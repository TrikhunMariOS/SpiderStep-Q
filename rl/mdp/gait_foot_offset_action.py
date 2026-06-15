# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Perceptive action term: RL residual in Cartesian foot space.

Unlike GaitResidualAction (residual in joint space), this adds the residual to the
foot target before IK:

    foot_target  = gait_foot_traj_with_cog(phase, cmd) + tanh(policy) * scale_mm
    joint_target = EMA( IK(foot_target) )

Foot space makes obstacle crossing (a foot-placement problem) much easier to learn.
Action channel 13 is a dynamic gait-frequency command. Settle, walk-ramp, EMA and
command conversion are the same as GaitResidualAction.

Sanity check: offset_scale_mm=0 -> behaves exactly like the proven gait baseline.
offset_mask (0,0,1) = height-only to start; (1,1,1) unlocks horizontal placement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

# robust import of the gait engine (works flat or as a package submodule)
try:
    from ..gait_torch import SpiderGaitEngine, JOINT_NAMES_FLAT, SWING_RATIO
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gait_torch import SpiderGaitEngine, JOINT_NAMES_FLAT, SWING_RATIO

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GaitFootOffsetAction(ActionTerm):
    """Residual foot-position (Cartesian) action on top of the procedural gait."""

    cfg: GaitFootOffsetActionCfg

    def __init__(self, cfg: GaitFootOffsetActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # map engine joint order -> articulation joint indices
        self._joint_ids, self._joint_names = self._asset.find_joints(
            JOINT_NAMES_FLAT, preserve_order=True
        )
        assert len(self._joint_ids) == 12, f"expected 12 leg joints, got {self._joint_names}"

        self._engine = SpiderGaitEngine(
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.float32,
            step_frequency=cfg.step_frequency,
            enable_cog_shift=cfg.enable_cog_shift,
        )

        # buffers
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._foot_offset = torch.zeros(self.num_envs, 4, 3, device=self.device)  # [E,4,3] mm
        # dynamic gait: phase is integrated (phase += freq*dt) so the RL frequency can
        # change mid-walk without the feet jumping.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)         # [E] cycles
        self._freq = torch.full((self.num_envs,), cfg.freq_base, device=self.device)  # [E] Hz
        self._swing_mask = torch.zeros(self.num_envs, 4, device=self.device)      # [E,4]
        self._swing_weight = torch.zeros(self.num_envs, 4, device=self.device)    # [E,4]
        self._swing_progress = torch.zeros(self.num_envs, 4, device=self.device)  # [E,4]
        # real foot vertical velocity (mm/s) from FK finite-difference, for the impact reward
        default_q = self._asset.data.default_joint_pos[:, self._joint_ids].reshape(-1, 4, 3)
        self._foot_z_prev = self._engine.fk(default_q)[..., 2].clone()            # [E,4] mm
        self._foot_vz = torch.zeros(self.num_envs, 4, device=self.device)         # [E,4] mm/s
        self._steps_since_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # EMA-smoothed joint target. Keep it: filters the fast swing so the underdamped
        # joints aren't whipped.
        self._gait_smooth = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # active foot-offset axes (Δx, Δy, Δz); (0,0,1) = height-only
        self._offset_mask = torch.tensor(cfg.offset_mask, device=self.device).view(1, 1, 3)  # [1,1,3]

        self._physics_dt = self._env.physics_dt

        # standstill gate: 0 = no command (RL blended out), 1 = walking
        self._stand_gate = torch.zeros(self.num_envs, 1, device=self.device)     # [E,1]
        self._stand_gate_alpha = self._physics_dt / max(cfg.stand_gate_tau_s, 1e-3)

        # command -> engine-unit conversion
        self._cmd_sign = torch.tensor(cfg.command_signs, device=self.device)   # [3]
        self._lin_scale = cfg.lin_vel_scale   # m/s  -> mm/s
        self._ang_scale = cfg.ang_vel_scale   # rad/s-> deg/s

        print(f"[GaitFootOffsetAction] offset_scale_mm={cfg.offset_scale_mm} "
              f"offset_mask={cfg.offset_mask} cog={cfg.enable_cog_shift}")
        print(f"[GaitFootOffsetAction] dynamic freq: base={cfg.freq_base}Hz "
              f"range=[{cfg.freq_base - cfg.freq_down:.1f}, {cfg.freq_base + cfg.freq_up:.1f}]Hz")
        print(f"[GaitFootOffsetAction] joints (engine order): {self._joint_names}")

    @property
    def action_dim(self) -> int:
        return 13   # 4 legs x (dx, dy, dz) + 1 gait-frequency channel

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def gait_phase_obs(self) -> torch.Tensor:
        """[E,2] = (sin, cos) of the global gait phase, for the gait_phase observation."""
        ang = 2.0 * math.pi * torch.remainder(self._gait_phase, 1.0)             # [E]
        return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)             # [E,2]

    @property
    def gait_freq(self) -> torch.Tensor:
        """[E] = currently executing (EMA-smoothed) gait frequency in Hz."""
        return self._freq

    @property
    def swing_mask(self) -> torch.Tensor:
        """[E,4] = 1.0 while each leg is swinging (0 during stance/settle)."""
        return self._swing_mask

    @property
    def swing_weight(self) -> torch.Tensor:
        """[E,4] = sin(pi * swing_progress): 0 at liftoff/touchdown, 1 at mid-swing."""
        return self._swing_weight

    @property
    def foot_offset_z(self) -> torch.Tensor:
        """[E,4] = the policy's commanded vertical foot offset Δz (mm)."""
        return self._foot_offset[..., 2]

    @property
    def swing_progress(self) -> torch.Tensor:
        """[E,4] = swing progress u in [0,1] (0=liftoff, 1=touchdown); 1 during stance."""
        return self._swing_progress

    @property
    def foot_vz(self) -> torch.Tensor:
        """[E,4] = real foot vertical velocity (mm/s, body frame, FK-based)."""
        return self._foot_vz

    def process_actions(self, actions: torch.Tensor):
        """Once per policy step: actions[:, :12] -> foot offsets (mm); actions[:, 12] -> gait freq."""
        self._raw_actions[:] = actions

        # foot offsets: tanh -> bounded, scale to ±offset_scale_mm, masked per axis
        off = torch.tanh(actions[:, :12]).reshape(self.num_envs, 4, 3) * self.cfg.offset_scale_mm
        off = off * self._offset_mask
        self._foot_offset = off

        # gait-frequency channel: zero action = base freq; +1 -> base+freq_up, -1 -> base-freq_down
        tf = torch.tanh(actions[:, 12])                                            # [E] in (-1,1)
        freq_cmd = (self.cfg.freq_base
                    + torch.relu(tf) * self.cfg.freq_up
                    - torch.relu(-tf) * self.cfg.freq_down)                        # [E] Hz
        a = self.cfg.freq_smoothing_alpha
        self._freq = (1.0 - a) * self._freq + a * freq_cmd

        self._processed_actions = torch.cat(
            [off.reshape(self.num_envs, 12), self._freq.unsqueeze(-1)], dim=-1
        )

    def apply_actions(self):
        """Every physics step: gait foot trajectory + RL foot offset -> IK -> joint targets."""
        # command in base frame (m/s, rad/s) -> engine units (mm/s, deg/s)
        cmd = self._env.command_manager.get_command(self.cfg.command_name)       # [E,3]
        cmd_eng = torch.empty_like(cmd)
        cmd_eng[:, 0] = cmd[:, 0] * self._lin_scale * self._cmd_sign[0]
        cmd_eng[:, 1] = cmd[:, 1] * self._lin_scale * self._cmd_sign[1]
        cmd_eng[:, 2] = cmd[:, 2] * self._ang_scale * self._cmd_sign[2]

        # SETTLE: hold the gait stance (phase frozen) for the first settle_steps
        settling = (self._steps_since_reset < self.cfg.settle_steps)                 # [E] bool

        # WALK RAMP: ease command + offset 0->1 after settle so it doesn't launch
        steps_after = (self._steps_since_reset - self.cfg.settle_steps).clamp(min=0).float()
        ramp = (steps_after / max(self.cfg.walk_ramp_steps, 1)).clamp(max=1.0).unsqueeze(-1)  # [E,1]

        cmd_eng = cmd_eng * ramp

        # cache per-leg swing state for the clearance rewards. Gate by commanded motion:
        # at standstill the swing windows are fictional, and ungated rewards trained the
        # policy to march in place.
        moving = (cmd_eng.abs().amax(dim=-1) > 5.0).float().unsqueeze(-1)        # [E,1]
        # standstill gate: blend RL authority out when there's no command (see stand_gate_tau_s)
        g = self._stand_gate_alpha
        self._stand_gate = (1.0 - g) * self._stand_gate + g * moving             # [E,1]
        leg_phase = self._engine.leg_phase_p(self._gait_phase)                   # [E,4] in [0,1)
        self._swing_mask = (leg_phase < SWING_RATIO).float() * (~settling).float().unsqueeze(-1) * moving
        # sine-bell swing weight: 0 at liftoff/touchdown, 1 mid-swing. The clearance reward
        # gates with this so the foot isn't punished for being near the ground at the endpoints.
        u = (leg_phase / SWING_RATIO).clamp(max=1.0)                             # [E,4] swing progress
        self._swing_weight = torch.sin(math.pi * u) * self._swing_mask
        self._swing_progress = u

        # real foot vertical velocity from FK finite-difference (mm/s), EMA-denoised
        foot_z_now = self._engine.fk(
            self._asset.data.joint_pos[:, self._joint_ids].reshape(-1, 4, 3)
        )[..., 2]                                                                # [E,4] mm
        vz_now = (foot_z_now - self._foot_z_prev) / self._physics_dt             # [E,4] mm/s
        self._foot_vz = 0.7 * self._foot_vz + 0.3 * vz_now
        self._foot_z_prev = foot_z_now

        # gait foot targets (mm, body frame) before IK. The walk ramp doubles as the CoG
        # ease-in. The RL frequency deviation and the CoG lean are gated by the standstill
        # gate so the body stands genuinely still at zero command.
        freq_eff = self.cfg.freq_base + (self._freq - self.cfg.freq_base) * self._stand_gate.squeeze(-1)
        period = (1.0 / freq_eff.clamp(min=0.1)).unsqueeze(-1)                   # [E,1] s
        foot = self._engine.foot_targets_with_cog_p(
            self._gait_phase, cmd_eng, period, ramp * self._stand_gate
        )

        # inject the RL foot offset (ramped at start, gated at standstill)
        foot = foot + self._foot_offset * ramp.unsqueeze(-1) * self._stand_gate.unsqueeze(-1)

        # IK: foot target -> joint angles  [E,12]
        q = self._engine.ik(foot).reshape(self.num_envs, 12)

        # EMA low-pass. alpha scales with frequency (constant smoothing per gait cycle): a
        # fixed lag eats a larger fraction of the swing at high freq. Capped for anti-whip.
        a = (self.cfg.joint_smoothing_alpha * freq_eff / self.cfg.freq_base).clamp(
            max=self.cfg.joint_smoothing_alpha_max
        ).unsqueeze(-1)                                                           # [E,1]
        self._gait_smooth = (1.0 - a) * self._gait_smooth + a * q
        target = self._gait_smooth
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)

        # advance the gait clock by the current per-env effective frequency (phase integration)
        self._gait_phase = self._gait_phase + (~settling).float() * freq_eff * self._physics_dt
        self._steps_since_reset += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._gait_phase[env_ids] = 0.0
        self._freq[env_ids] = self.cfg.freq_base
        self._stand_gate[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._foot_offset[env_ids] = 0.0
        self._swing_mask[env_ids] = 0.0
        self._swing_weight[env_ids] = 0.0
        self._swing_progress[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        self._gait_smooth[env_ids] = self._asset.data.default_joint_pos[:, self._joint_ids][env_ids]
        # re-seed the FK finite-difference so the first vz after reset isn't a spike
        default_q = self._asset.data.default_joint_pos[:, self._joint_ids][env_ids].reshape(-1, 4, 3)
        self._foot_z_prev[env_ids] = self._engine.fk(default_q)[..., 2]
        self._foot_vz[env_ids] = 0.0


@configclass
class GaitFootOffsetActionCfg(ActionTermCfg):
    """Configuration for :class:`GaitFootOffsetAction` (replaces the env's joint_pos term)."""

    class_type: type[ActionTerm] = GaitFootOffsetAction
    asset_name: str = MISSING

    # RL foot offset (Cartesian, body frame, mm), ±offset_scale_mm per active axis.
    # Mainly for stance-leveling + extra clearance on top of the gait's swing lift.
    offset_scale_mm: float = 40.0
    # active axes (Δx, Δy, Δz); (0,0,1) = height-only to start, (1,1,1) unlocks placement
    offset_mask: tuple[float, float, float] = (0.0, 0.0, 1.0)

    step_frequency: float = 1.2          # Hz; engine default for fixed-freq wrappers/tests
    enable_cog_shift: bool = True

    # dynamic gait frequency (action channel 13):
    #   freq = freq_base + relu(tanh(a))*freq_up - relu(-tanh(a))*freq_down
    #   zero action -> freq_base (proven safe default), +1 -> fast, -1 -> careful crawl
    freq_base: float = 1.2               # Hz at zero action
    freq_up: float = 1.3                 # Hz of headroom above base
    freq_down: float = 0.4               # Hz of headroom below base
    freq_smoothing_alpha: float = 0.2    # EMA on the frequency command

    command_name: str = "base_velocity"

    lin_vel_scale: float = 1000.0        # m/s  -> mm/s
    ang_vel_scale: float = 180.0 / math.pi   # rad/s -> deg/s
    command_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)

    settle_steps: int = 120              # physics steps held at neutral stance after reset
    walk_ramp_steps: int = 100           # ease command + offset 0->full over this many steps
    joint_smoothing_alpha: float = 0.12  # EMA on the joint target at freq_base
    # alpha scales with frequency (constant smoothing per cycle) since the lag is fixed in
    # seconds; capped to keep the anti-whip protection on the underdamped joints.
    joint_smoothing_alpha_max: float = 0.30

    # standstill gate: at ~zero command the RL offsets and freq deviation are blended out
    # over this time constant so the robot adds nothing and doesn't fidget/drift.
    stand_gate_tau_s: float = 0.2
