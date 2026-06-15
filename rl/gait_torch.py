# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized (batched) torch port of the test5 gait + IK.

Reproduces the exact math of OrientationCompensation_test5.py (the numpy reference)
but batched over [num_envs, 4 legs] on the GPU, with a per-env command and gait clock.

Correctness contract (verified by test_ik_validation.py): float64 matches the numpy
reference to < 1e-6 rad; float32 (training dtype) to < ~1e-4 rad. Units: mm in, rad out.

Internally shaped [E, 4, 3] (leg axis = LEGS, joint axis = coxa/femur/tibia).
joint_targets() returns a flat [E, 12] in JOINT_NAMES_FLAT order; the action term maps
these names to articulation joint indices via find_joints().
"""

from __future__ import annotations

import math
import torch

# import works both flat (standalone validation: `import gait_torch`) and as a
# package submodule (RL task: `from ..gait_torch import ...`)
try:
    from gait_numpy_ref import (
        L1, L2, L3,
        LEGS, LEG_SIGN, HIP_POS_BODY, ZERO_FOOT_BODY, GAIT_FOOT_BODY, JOINT_LIMITS,
        STEP_FREQUENCY, STEP_HEIGHT, MAX_STRIDE_LENGTH, TURN_GAIN, SWING_RATIO,
        BEZIER_LIFT_FRONT_RATIO, BEZIER_LIFT_BACK_RATIO, BEZIER_LAND_RATIO,
        PHASE_OFFSET, COG_SHIFT_GAIN, COG_LEAD, COG_RAMP_TIME, COG_MAX_OFFSET_MM,
        STANCE_RAMP,
    )
except ImportError:  # imported as part of the rl package
    from .gait_numpy_ref import (
        L1, L2, L3,
        LEGS, LEG_SIGN, HIP_POS_BODY, ZERO_FOOT_BODY, GAIT_FOOT_BODY, JOINT_LIMITS,
        STEP_FREQUENCY, STEP_HEIGHT, MAX_STRIDE_LENGTH, TURN_GAIN, SWING_RATIO,
        BEZIER_LIFT_FRONT_RATIO, BEZIER_LIFT_BACK_RATIO, BEZIER_LAND_RATIO,
        PHASE_OFFSET, COG_SHIFT_GAIN, COG_LEAD, COG_RAMP_TIME, COG_MAX_OFFSET_MM,
        STANCE_RAMP,
    )

JOINTS = ["coxa", "femur", "tibia"]
JOINT_NAMES_FLAT = [f"{j}_{leg}_joint" for leg in LEGS for j in JOINTS]
# -> ['coxa_FL_joint','femur_FL_joint','tibia_FL_joint','coxa_FR_joint', ...]

_EPS = 1e-6


def _smoothstep(u: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class SpiderGaitEngine:
    """Batched, differentiable-friendly procedural gait + analytic IK.

    Parameters
    ----------
    num_envs : int
    device   : torch.device | str
    dtype    : torch.dtype   (use float64 for the validation, float32 for training)
    step_frequency : Hz; default = test5's 1.2
    """

    def __init__(
        self,
        num_envs: int,
        device="cpu",
        dtype: torch.dtype = torch.float32,
        step_frequency: float = STEP_FREQUENCY,
        enable_cog_shift: bool = True,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.freq = float(step_frequency)
        self.enable_cog = enable_cog_shift

        tk = dict(device=self.device, dtype=self.dtype)

        # --- per-leg geometry, shape [4, 3] / [4] in LEGS order -------------
        self.hip_pos   = torch.tensor([HIP_POS_BODY[l]   for l in LEGS], **tk)   # [4,3]
        self.zero_foot = torch.tensor([ZERO_FOOT_BODY[l] for l in LEGS], **tk)   # [4,3]
        self.gait_foot = torch.tensor([GAIT_FOOT_BODY[l] for l in LEGS], **tk)   # [4,3]
        self.leg_sign  = torch.tensor(
            [[LEG_SIGN[l]["coxa"], LEG_SIGN[l]["femur"], LEG_SIGN[l]["tibia"]] for l in LEGS],
            **tk,
        )  # [4,3]
        self.phase_offset = torch.tensor([PHASE_OFFSET[l] for l in LEGS], **tk)  # [4]

        # joint limits [4,3] low / high
        self.lim_lo = torch.tensor(
            [[JOINT_LIMITS[l]["coxa"][0], JOINT_LIMITS[l]["femur"][0], JOINT_LIMITS[l]["tibia"][0]] for l in LEGS],
            **tk,
        )
        self.lim_hi = torch.tensor(
            [[JOINT_LIMITS[l]["coxa"][1], JOINT_LIMITS[l]["femur"][1], JOINT_LIMITS[l]["tibia"][1]] for l in LEGS],
            **tk,
        )

        # --- ik-plane base angle per leg  (atan2 of zero-foot in leg frame) -
        zf_leg = self.zero_foot - self.hip_pos              # [4,3]
        base_angle = torch.atan2(zf_leg[:, 1], zf_leg[:, 0])  # [4]
        # test5 uses c = cos(-base), s = sin(-base)
        self.plane_c = torch.cos(-base_angle)               # [4]
        self.plane_s = torch.sin(-base_angle)               # [4]

        # --- neutral joint angles [4,3] (theta1,2,3 at the zero foot) -------
        # run the SAME ik on the zero foot, no sign / no neutral subtraction
        zf = self.zero_foot.unsqueeze(0)                    # [1,4,3]
        n1, n2, n3 = self._raw_ik(*self._to_ik_plane(zf))   # each [1,4]
        self.neutral = torch.stack([n1, n2, n3], dim=-1).squeeze(0)  # [4,3]

        # scalar constants as tensors
        self.gait_period = 1.0 / max(self.freq, _EPS)

    # ------------------------------------------------------------------ #
    #  IK  (batched, knee = "down")                                      #
    # ------------------------------------------------------------------ #
    def _to_ik_plane(self, foot_body_mm: torch.Tensor):
        """[E,4,3] body-frame foot -> (x,y,z) each [E,4] in the per-leg IK plane."""
        leg = foot_body_mm - self.hip_pos                   # broadcast [E,4,3]
        x, y, z = leg[..., 0], leg[..., 1], leg[..., 2]     # [E,4]
        c, s = self.plane_c, self.plane_s                   # [4]
        xp = c * x - s * y
        yp = s * x + c * y
        return xp, yp, z

    def _raw_ik(self, x, y, z):
        """Inputs each [...,4].  Returns theta1, theta2, theta3 (knee='down')."""
        theta1 = torch.atan2(y, x)
        horizontal = torch.sqrt(x * x + y * y)
        r = horizontal - L1
        D = torch.sqrt(r * r + z * z)

        safe_min = abs(L2 - L3) + 5.0
        safe_max = L2 + L3 - 5.0

        # clamp D into [safe_min, safe_max] by SCALING r and z (matches test5)
        scale = torch.ones_like(D)
        scale = torch.where(D > safe_max, safe_max / torch.clamp(D, min=_EPS), scale)
        scale = torch.where(D < safe_min, safe_min / torch.clamp(D, min=_EPS), scale)
        r = r * scale
        z = z * scale
        D = torch.clamp(D, min=safe_min, max=safe_max)

        cos_alpha = (L2 ** 2 + D * D - L3 ** 2) / (2 * L2 * D)
        cos_beta  = (L2 ** 2 + L3 ** 2 - D * D) / (2 * L2 * L3)
        alpha = torch.acos(torch.clamp(cos_alpha, -1.0, 1.0))
        beta  = torch.acos(torch.clamp(cos_beta,  -1.0, 1.0))
        base  = torch.atan2(z, r)

        theta2 = base + alpha
        theta3 = math.pi - beta
        return theta1, theta2, theta3

    def ik(self, foot_body_mm: torch.Tensor) -> torch.Tensor:
        """[E,4,3] body-frame foot targets (mm) -> joint targets [E,4,3] (rad),
        sign + neutral applied + joint-limit clamped.  Identical to test5's
        `clamp_joint_leg(spider_ik_body(...))`."""
        t1, t2, t3 = self._raw_ik(*self._to_ik_plane(foot_body_mm))   # [E,4]
        theta = torch.stack([t1, t2, t3], dim=-1)                     # [E,4,3]
        q = self.leg_sign * (theta - self.neutral)                    # [E,4,3]
        q = torch.clamp(q, self.lim_lo, self.lim_hi)
        return q

    def fk(self, q: torch.Tensor) -> torch.Tensor:
        """Forward kinematics: [E,4,3] joint angles (rad) -> foot-CENTER positions [E,4,3]
        (mm, body frame). Exact inverse of :meth:`ik`.

        Feed it the real sim joint positions to get where the feet actually are — the
        trusted substitute for body_pos_w["Foot_*"], whose USD link frames are offset.
        """
        # undo sign + neutral:  q = sign*(theta - neutral)  ->  theta = neutral + q*sign
        theta = self.neutral + q * self.leg_sign            # [E,4,3]  (sign is ±1)
        t1, t2, t3 = theta[..., 0], theta[..., 1], theta[..., 2]   # each [E,4]

        # planar 2-link reach (femur + tibia) from the coxa axis
        r = L1 + L2 * torch.cos(t2) + L3 * torch.cos(t2 - t3)      # [E,4]
        z = L2 * torch.sin(t2) + L3 * torch.sin(t2 - t3)           # [E,4]

        # coxa yaw inside the IK plane
        x_local = r * torch.cos(t1)
        y_local = r * torch.sin(t1)

        # rotate back from the IK plane to the leg frame (+base_angle).
        # plane_c/plane_s were built with cos(-base)/sin(-base), so
        # cos(base) = plane_c and sin(base) = -plane_s.
        c, s = self.plane_c, -self.plane_s                   # [4]
        x_leg = c * x_local - s * y_local
        y_leg = s * x_local + c * y_local

        foot_leg = torch.stack([x_leg, y_leg, z], dim=-1)    # [E,4,3]
        return self.hip_pos + foot_leg                       # + hip -> body frame

    # ------------------------------------------------------------------ #
    #  GAIT TRAJECTORY  (batched)                                        #
    # ------------------------------------------------------------------ #
    # PHASE-BASED CORE (dynamic gait). The *_p methods take an accumulated global phase
    # (cycles, [E]) + per-env period [E,1] (=1/freq), so the caller can integrate
    # phase += freq*dt with a time-varying per-env frequency. The fixed-freq time-based
    # wrappers below delegate to these (bit-identical; verified by test_phase_refactor.py).

    def leg_phase_p(self, phase: torch.Tensor) -> torch.Tensor:
        """phase [E] global gait phase (cycles) -> per-leg phase [E,4] in [0,1)."""
        return torch.remainder(phase.unsqueeze(-1) + self.phase_offset, 1.0)

    def _leg_phase(self, t: torch.Tensor) -> torch.Tensor:
        """t [E] gait time (s) -> per-leg phase [E,4] in [0,1).  (fixed-freq wrapper)"""
        return self.leg_phase_p(t * self.freq)

    def _stride_vector_p(self, cmd: torch.Tensor, period: torch.Tensor):
        """cmd [E,3] (mm/s, mm/s, deg/s), period [E,1] (s) -> sx,sy each [E,1]."""
        sx = -cmd[:, 0:1] * period
        sy = -cmd[:, 1:2] * period
        mag = torch.sqrt(sx * sx + sy * sy)
        scale = torch.where(
            mag > MAX_STRIDE_LENGTH,
            MAX_STRIDE_LENGTH / torch.clamp(mag, min=_EPS),
            torch.ones_like(mag),
        )
        return sx * scale, sy * scale                       # [E,1] each

    def _stride_vector(self, cmd: torch.Tensor):
        """Fixed-period wrapper around :meth:`_stride_vector_p`."""
        period = torch.full((cmd.shape[0], 1), self.gait_period, device=cmd.device, dtype=cmd.dtype)
        return self._stride_vector_p(cmd, period)

    def _turn_offset_p(self, cmd: torch.Tensor, period: torch.Tensor):
        """Per-leg yaw-induced foot offset -> tx,ty each [E,4].  period [E,1]."""
        wz_deg = cmd[:, 2:3]                                # [E,1]
        wz = torch.deg2rad(wz_deg)                          # [E,1]
        x0 = self.gait_foot[:, 0].unsqueeze(0)              # [1,4]
        y0 = self.gait_foot[:, 1].unsqueeze(0)              # [1,4]
        tx = -wz * y0 * period * TURN_GAIN                  # [E,4]
        ty = wz * x0 * period * TURN_GAIN                   # [E,4]
        active = (torch.abs(wz_deg) >= _EPS)                # [E,1]
        tx = torch.where(active, tx, torch.zeros_like(tx))
        ty = torch.where(active, ty, torch.zeros_like(ty))
        return tx, ty

    def _turn_offset(self, cmd: torch.Tensor):
        """Fixed-period wrapper around :meth:`_turn_offset_p`."""
        period = torch.full((cmd.shape[0], 1), self.gait_period, device=cmd.device, dtype=cmd.dtype)
        return self._turn_offset_p(cmd, period)

    def foot_targets_p(self, gphase: torch.Tensor, cmd: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
        """gphase [E] (cycles), cmd [E,3], period [E,1] -> gait foot targets [E,4,3] (mm),
        BEFORE the CoG shift.  Phase-based core — supports per-env, time-varying frequency."""
        E = gphase.shape[0]
        phase = self.leg_phase_p(gphase)                    # [E,4]
        x0 = self.gait_foot[:, 0].unsqueeze(0).expand(E, 4)  # [E,4]
        y0 = self.gait_foot[:, 1].unsqueeze(0).expand(E, 4)
        z0 = self.gait_foot[:, 2].unsqueeze(0).expand(E, 4)

        sx, sy = self._stride_vector_p(cmd, period)         # [E,1]
        tdx, tdy = self._turn_offset_p(cmd, period)         # [E,4]
        dx = sx + tdx                                       # [E,4]
        dy = sy + tdy

        moving = (torch.sqrt(dx * dx + dy * dy) >= _EPS)    # [E,4]

        xb, yb = x0 - dx / 2, y0 - dy / 2
        xf, yf = x0 + dx / 2, y0 + dy / 2

        # ---- swing branch (phase < SWING_RATIO): cubic bezier ----
        u_sw = torch.clamp(phase / SWING_RATIO, 0.0, 1.0)
        # control points (x,y,z)
        p0 = torch.stack([xb, yb, z0], dim=-1)
        p1 = torch.stack([xb + (xf - xb) * BEZIER_LIFT_FRONT_RATIO,
                          yb + (yf - yb) * BEZIER_LIFT_FRONT_RATIO,
                          z0 + STEP_HEIGHT], dim=-1)
        p2 = torch.stack([xb + (xf - xb) * BEZIER_LIFT_BACK_RATIO,
                          yb + (yf - yb) * BEZIER_LIFT_BACK_RATIO,
                          z0 + STEP_HEIGHT * BEZIER_LAND_RATIO], dim=-1)
        p3 = torch.stack([xf, yf, z0], dim=-1)
        uu = u_sw.unsqueeze(-1)
        vv = 1.0 - uu
        swing = (vv ** 3) * p0 + 3 * (vv ** 2) * uu * p1 \
            + 3 * vv * (uu ** 2) * p2 + (uu ** 3) * p3      # [E,4,3]

        # ---- stance branch: smoothstep slide back ----
        u_st = torch.clamp((phase - SWING_RATIO) / (1.0 - SWING_RATIO), 0.0, 1.0)
        us = _smoothstep(u_st)
        st_x = xf + (xb - xf) * us
        st_y = yf + (yb - yf) * us
        stance = torch.stack([st_x, st_y, z0], dim=-1)      # [E,4,3]

        is_swing = (phase < SWING_RATIO).unsqueeze(-1)      # [E,4,1]
        traj = torch.where(is_swing, swing, stance)         # [E,4,3]

        # ---- if no commanded motion, foot stays at nominal (no in-place lift)
        nominal = torch.stack([x0, y0, z0], dim=-1)         # [E,4,3]
        traj = torch.where(moving.unsqueeze(-1), traj, nominal)
        return traj

    def foot_targets(self, t: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        """Fixed-frequency wrapper: t [E] (s) -> foot targets, identical to the old API."""
        period = torch.full((cmd.shape[0], 1), self.gait_period, device=cmd.device, dtype=cmd.dtype)
        return self.foot_targets_p(t * self.freq, cmd, period)

    # ------------------------------------------------------------------ #
    #  FEEDFORWARD CoG SHIFT  (batched)                                  #
    # ------------------------------------------------------------------ #
    def cog_offset_p(self, gphase: torch.Tensor, ramp: torch.Tensor) -> torch.Tensor:
        """gphase [E] (cycles), ramp [E,1] in [0,1] -> body lean (cog_x, cog_y) [E,2] (mm).

        Phase-based core.  `ramp` replaces the old t/COG_RAMP_TIME ease-in (the caller
        owns the notion of time-since-walk-start; the engine no longer needs wall time).
        """
        # phase per leg, led by COG_LEAD
        p = torch.remainder(gphase.unsqueeze(-1) + self.phase_offset + COG_LEAD, 1.0)  # [E,4]
        w = self._stance_weight(p)                          # [E,4]
        x = self.gait_foot[:, 0].unsqueeze(0)               # [1,4]
        y = self.gait_foot[:, 1].unsqueeze(0)
        wsum = w.sum(dim=-1, keepdim=True)                  # [E,1]
        cx = (w * x).sum(dim=-1, keepdim=True)              # [E,1]
        cy = (w * y).sum(dim=-1, keepdim=True)

        safe = wsum > _EPS
        tx = torch.where(safe, ramp * COG_SHIFT_GAIN * (cx / torch.clamp(wsum, min=_EPS)),
                         torch.zeros_like(cx))
        ty = torch.where(safe, ramp * COG_SHIFT_GAIN * (cy / torch.clamp(wsum, min=_EPS)),
                         torch.zeros_like(cy))

        mag = torch.sqrt(tx * tx + ty * ty)
        clamp_scale = torch.where(
            mag > COG_MAX_OFFSET_MM,
            COG_MAX_OFFSET_MM / torch.clamp(mag, min=_EPS),
            torch.ones_like(mag),
        )
        tx = tx * clamp_scale
        ty = ty * clamp_scale
        return torch.cat([tx, ty], dim=-1)                  # [E,2]

    def cog_offset(self, t: torch.Tensor) -> torch.Tensor:
        """Fixed-frequency wrapper: t [E] (s) -> CoG lean, identical to the old API."""
        ramp = torch.clamp(t.unsqueeze(-1) / max(COG_RAMP_TIME, _EPS), 0.0, 1.0)  # [E,1]
        return self.cog_offset_p(t * self.freq, ramp)

    def _stance_weight(self, p: torch.Tensor) -> torch.Tensor:
        """Smooth support weight in [0,1].  0 during swing, 1 mid-stance."""
        p = torch.remainder(p, 1.0)
        up   = _smoothstep((p - SWING_RATIO) / STANCE_RAMP)
        down = _smoothstep((1.0 - p) / STANCE_RAMP)
        w = torch.minimum(up, down)
        return torch.where(p < SWING_RATIO, torch.zeros_like(w), w)

    # ------------------------------------------------------------------ #
    #  FULL PIPELINE                                                     #
    # ------------------------------------------------------------------ #
    def foot_targets_with_cog_p(
        self, gphase: torch.Tensor, cmd: torch.Tensor, period: torch.Tensor, cog_ramp: torch.Tensor
    ) -> torch.Tensor:
        """Phase-based core: gphase [E], cmd [E,3], period [E,1], cog_ramp [E,1]
        -> gait foot targets AFTER the feedforward CoG shift [E,4,3] (mm)."""
        foot = self.foot_targets_p(gphase, cmd, period)     # [E,4,3]
        if self.enable_cog:
            cog = self.cog_offset_p(gphase, cog_ramp)       # [E,2]
            shift = torch.zeros_like(foot)
            shift[..., 0] = -cog[:, 0:1]                    # subtract cog_x from all legs
            shift[..., 1] = -cog[:, 1:2]
            foot = foot + shift
        return foot

    def foot_targets_with_cog(self, t: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        """[E,4,3] gait foot targets AFTER the feedforward CoG shift (mm).  (fixed-freq wrapper)"""
        period = torch.full((cmd.shape[0], 1), self.gait_period, device=cmd.device, dtype=cmd.dtype)
        cog_ramp = torch.clamp(t.unsqueeze(-1) / max(COG_RAMP_TIME, _EPS), 0.0, 1.0)
        return self.foot_targets_with_cog_p(t * self.freq, cmd, period, cog_ramp)

    def joint_targets(self, t: torch.Tensor, cmd: torch.Tensor, flat: bool = True):
        """Full test5 pipeline: gait time + per-env command -> joint targets.

        t   : [E]   gait clock (s)
        cmd : [E,3] = (vx_mm_s, vy_mm_s, wz_deg_s)
        flat: True  -> [E,12] in JOINT_NAMES_FLAT order (leg-major coxa/femur/tibia)
              False -> [E,4,3] (leg, joint)
        """
        foot = self.foot_targets_with_cog(t, cmd)           # [E,4,3]
        q = self.ik(foot)                                   # [E,4,3]
        return q.reshape(q.shape[0], 12) if flat else q
