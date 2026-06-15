# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward terms for the residual / foot-offset gait policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_swing_suppression(env: ManagerBasedRLEnv, action_term_name: str = "joint_pos") -> torch.Tensor:
    """Penalty for pushing a foot DOWN (negative Δz) while that leg should be swinging.

    Stops the degenerate gait where the policy lowers bad_orientation by not lifting the
    rear legs. Only penalizes downward offset during swing. Returns a positive penalty in
    ~[0,1] (use a NEGATIVE weight).
    """
    term = env.action_manager._terms[action_term_name]
    swing = term.swing_mask                                   # [E,4]
    dz = term.foot_offset_z                                   # [E,4] mm
    scale = max(float(term.cfg.offset_scale_mm), 1e-6)        # normalize -> [0,1]
    push_down = torch.clamp(-dz, min=0.0) / scale             # [E,4] only the downward part
    num = (swing * push_down ** 2).sum(dim=-1)                # [E]
    den = swing.sum(dim=-1).clamp(min=1e-6)                   # [E] (# swinging legs)
    return num / den                                          # [E] mean over swing legs, in [0,1]


def foot_clearance_over_terrain(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
    sensor_name: str = "height_scanner",
    target_clearance: float = 0.04,     # m — desired swing-foot height above local terrain
    foot_radius: float = 0.010,         # m — FK tracks the foot-ball center; contact is 10mm lower
    neighborhood_radius: float = 0.06,  # m — clear the highest ground within this xy radius
) -> torch.Tensor:
    """Swing-foot clearance measured against the terrain under each foot.

    The perception reward: on a raised box the required lift rises too, and height_scan is
    the only obs that reveals boxes, so maximizing this forces the policy to read it.
    Gated by swing_mask. Returns a positive penalty (use a NEGATIVE weight).
    """
    term = env.action_manager._terms[action_term_name]
    asset = env.scene["robot"]

    # real foot positions from FK on measured joint angles  [E,4,3] m, body frame
    q = asset.data.joint_pos[:, term._joint_ids].reshape(-1, 4, 3)        # [E,4,3] rad
    foot_b = term._engine.fk(q) / 1000.0                                  # mm -> m

    # body frame -> world frame with the root pose
    E = foot_b.shape[0]
    quat = asset.data.root_quat_w.repeat_interleave(4, dim=0)             # [E*4, 4]
    foot_w = quat_apply(quat, foot_b.reshape(E * 4, 3)).reshape(E, 4, 3)
    foot_w = foot_w + asset.data.root_pos_w.unsqueeze(1)                  # [E,4,3]

    # terrain height to clear = max scanner-hit z within neighborhood_radius of the foot,
    # so an upcoming box top becomes the target before the toe reaches its face.
    if sensor_name is None or sensor_name not in env.scene.sensors:
        # flat arena: ground is the plane at z=0 (same reward keeps the gait clean on flat)
        ground_z = torch.zeros_like(foot_w[..., 2])                       # [E,4]
    else:
        hits = env.scene.sensors[sensor_name].data.ray_hits_w             # [E,N,3]
        hits_z = hits[..., 2]                                             # [E,N]
        # rays that miss return inf/nan -> exclude them
        bad = ~torch.isfinite(hits_z)                                     # [E,N]
        hits_z = torch.where(bad, torch.zeros_like(hits_z), hits_z)
        d_xy = (hits[..., :2].unsqueeze(1) - foot_w[..., :2].unsqueeze(2)).square().sum(-1)  # [E,4,N]
        d_xy = d_xy + bad.unsqueeze(1) * 1.0e6
        hits_z_e = hits_z.unsqueeze(1).expand(-1, 4, -1)                  # [E,4,N]
        # max ground height within the radius...
        within = d_xy <= neighborhood_radius ** 2                         # [E,4,N]
        neigh = torch.where(within, hits_z_e, torch.full_like(hits_z_e, -1.0e6))
        ground_max = neigh.max(dim=-1).values                             # [E,4]
        # ...with nearest-hit fallback if no ray fell inside the radius
        nearest = d_xy.argmin(dim=-1)                                     # [E,4]
        ground_near = torch.gather(hits_z_e, 2, nearest.unsqueeze(-1)).squeeze(-1)
        ground_z = torch.where(ground_max > -1.0e5, ground_max, ground_near)  # [E,4]

    # clearance of the foot contact point above the local ground
    clearance = foot_w[..., 2] - foot_radius - ground_z                   # [E,4] m

    # penalize the shortfall, gated by the sine-bell swing weight so clearance is only
    # required mid-swing (not at the near-ground liftoff/touchdown endpoints)
    w = term.swing_weight                                                 # [E,4]
    shortfall = torch.clamp(target_clearance - clearance, min=0.0) / target_clearance
    num = (w * shortfall ** 2).sum(dim=-1)                                # [E]
    den = w.sum(dim=-1).clamp(min=1e-6)
    return num / den                                                      # [E] in [0,1]


def foot_touchdown_impact(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
    window_start: float = 0.6,       # late-swing window: penalty active from 60% of swing
    vz_norm_mm_s: float = 500.0,     # descent speed (mm/s) that counts as penalty = 1.0
) -> torch.Tensor:
    """Penalty for slamming a foot down (fast descent in the late swing window).

    Asks for a soft, braked touchdown; early/mid swing stays free so fast lifting over
    obstacles stays cheap. Returns a positive penalty (use a NEGATIVE weight).
    """
    term = env.action_manager._terms[action_term_name]
    u = term.swing_progress                                               # [E,4] 0..1
    swing = term.swing_mask                                               # [E,4]

    # late-swing window weight: 0 before window_start, smoothstep up to 1 at touchdown
    w = ((u - window_start) / max(1.0 - window_start, 1e-6)).clamp(0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w) * swing                                   # [E,4]

    down = torch.clamp(-term.foot_vz, min=0.0) / vz_norm_mm_s             # [E,4] descending only
    num = (w * down ** 2).sum(dim=-1)                                     # [E]
    den = w.sum(dim=-1).clamp(min=1e-6)
    return num / den                                                      # [E]
