# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Residual RL action term.

Turns the policy output into a small residual on top of the procedural gait:

    joint_target = procedural_gait(phase, command) + tanh(policy) * residual_scale

The gait baseline already walks (see play_gait_only.py); the policy only learns a
small correction, so it trains fast and stays stable.
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
    from ..gait_torch import SpiderGaitEngine, JOINT_NAMES_FLAT
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gait_torch import SpiderGaitEngine, JOINT_NAMES_FLAT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GaitResidualAction(ActionTerm):
    """Residual joint-position action on top of the procedural gait."""

    cfg: GaitResidualActionCfg

    def __init__(self, cfg: GaitResidualActionCfg, env: ManagerBasedRLEnv):
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
        self._phase_time = torch.zeros(self.num_envs, device=self.device)   # per-env gait clock (s)
        self._steps_since_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # EMA-smoothed gait target. Keep it: filtering the fast swing is what stops the
        # underdamped joints from being whipped (robot launches/tips without it).
        self._gait_smooth = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._physics_dt = self._env.physics_dt

        # command -> engine-unit conversion
        self._cmd_sign = torch.tensor(cfg.command_signs, device=self.device)   # [3]
        self._lin_scale = cfg.lin_vel_scale   # m/s  -> mm/s
        self._ang_scale = cfg.ang_vel_scale   # rad/s-> deg/s

        print(f"[GaitResidualAction] residual_scale={cfg.residual_scale} "
              f"freq={cfg.step_frequency} cog={cfg.enable_cog_shift}")
        print(f"[GaitResidualAction] joints (engine order): {self._joint_names}")

    @property
    def action_dim(self) -> int:
        return 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def gait_phase_obs(self) -> torch.Tensor:
        """[E,2] = (sin, cos) of the global gait phase, for the gait_phase observation."""
        base_phase = torch.remainder(self._phase_time * self._engine.freq, 1.0)  # [E]
        ang = 2.0 * math.pi * base_phase
        return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)             # [E,2]

    def process_actions(self, actions: torch.Tensor):
        """Once per policy step: store raw output, squash to a bounded residual."""
        self._raw_actions[:] = actions
        self._processed_actions = torch.tanh(actions) * self.cfg.residual_scale

    def apply_actions(self):
        """Every physics step: gait baseline (advancing clock) + residual -> joint targets."""
        # command in base frame (m/s, rad/s) -> engine units (mm/s, deg/s)
        cmd = self._env.command_manager.get_command(self.cfg.command_name)       # [E,3]
        cmd_eng = torch.empty_like(cmd)
        cmd_eng[:, 0] = cmd[:, 0] * self._lin_scale * self._cmd_sign[0]
        cmd_eng[:, 1] = cmd[:, 1] * self._lin_scale * self._cmd_sign[1]
        cmd_eng[:, 2] = cmd[:, 2] * self._ang_scale * self._cmd_sign[2]

        # SETTLE: hold the gait stance (phase frozen) for the first settle_steps so the
        # robot drops onto its feet before walking.
        settling = (self._steps_since_reset < self.cfg.settle_steps)                 # [E] bool

        # WALK RAMP: ease command + residual 0->1 after settle so the robot doesn't launch.
        steps_after = (self._steps_since_reset - self.cfg.settle_steps).clamp(min=0).float()
        ramp = (steps_after / max(self.cfg.walk_ramp_steps, 1)).clamp(max=1.0).unsqueeze(-1)  # [E,1]

        cmd_eng = cmd_eng * ramp

        # procedural gait baseline at the current phase  [E,12]
        q_gait = self._engine.joint_targets(self._phase_time, cmd_eng, flat=True)

        # EMA low-pass the gait baseline (see _gait_smooth note)
        a = self.cfg.joint_smoothing_alpha
        self._gait_smooth = (1.0 - a) * self._gait_smooth + a * q_gait

        # add the ramped residual and command the joints
        target = self._gait_smooth + self._processed_actions * ramp
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)

        # advance the gait clock only for envs that finished settling
        self._phase_time = self._phase_time + (~settling).float() * self._physics_dt
        self._steps_since_reset += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._phase_time[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        self._gait_smooth[env_ids] = self._asset.data.default_joint_pos[:, self._joint_ids][env_ids]


@configclass
class GaitResidualActionCfg(ActionTermCfg):
    """Configuration for :class:`GaitResidualAction` (replaces the env's joint_pos term)."""

    class_type: type[ActionTerm] = GaitResidualAction
    asset_name: str = MISSING

    residual_scale: float = 0.2          # rad; ±0.2 ≈ ±11.5°

    step_frequency: float = 1.2          # Hz
    enable_cog_shift: bool = True

    command_name: str = "base_velocity"

    # engine speaks test5 units (mm/s, deg/s, body frame front=-X); Isaac is m/s, rad/s,
    # base frame +X. Scales are fixed; flip a command_sign if the robot moves the wrong way.
    lin_vel_scale: float = 1000.0        # m/s  -> mm/s
    ang_vel_scale: float = 180.0 / math.pi   # rad/s -> deg/s
    command_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # hold neutral stance after reset so the robot settles on its feet (80 × dt0.0025 = 0.2 s)
    settle_steps: int = 80

    # ease command + residual in after settle so it doesn't launch (100 × dt0.0025 = 0.25 s)
    walk_ramp_steps: int = 100

    # EMA on the gait target. Keep near 0.12 — required to filter the fast swing or the
    # underdamped joints get whipped and the robot tips.
    joint_smoothing_alpha: float = 0.12
