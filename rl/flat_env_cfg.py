# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-terrain residual-gait velocity task (the speed arena).

On flat ground fast is both safe and rewarded, so this is where the dynamic gait's
frequency channel actually gets used. Versus the rough cfg: wider forward command,
sharper tracking reward, and lighter high-frequency penalties. The rough policy stays
the careful obstacle-crosser; this one is the sprinter.
"""

from isaaclab.utils import configclass

from .rough_env_cfg import SpiderResidualRoughEnvCfg


@configclass
class SpiderResidualFlatEnvCfg(SpiderResidualRoughEnvCfg):
    """Residual-gait SPEED training on a flat plane."""

    def __post_init__(self):
        super().__post_init__()

        # flat plane, no height scan, no terrain curriculum
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # keep swing-clearance ON (sensor_name=None -> measures over the z=0 plane);
        # removing it brought back the body-low / rear-drag degenerate gait on flat
        self.rewards.foot_clearance_terrain.params["sensor_name"] = None
        self.rewards.foot_clearance_terrain.weight = -1.0   # lighter than rough's -2.0

        # speed command envelope: vx to ±0.25 (dynamic-gait cap = 2.5 Hz × 100 mm),
        # so reaching it requires raising the gait frequency
        self.commands.base_velocity.ranges.lin_vel_x = (-0.25, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.12, 0.12)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)

        # sharpen velocity tracking so under-tracking isn't free -> upshifting becomes worth it
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.20
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.30
        self.rewards.track_lin_vel_xy_exp.weight = 2.0

        # relax the high-frequency taxes (less anti-jitter need on flat)
        self.rewards.dof_acc_l2.weight = -1.0e-7            # was -2.5e-7
        self.rewards.action_rate_l2.weight = -0.04          # was -0.1

        # flat training can lean harder on staying level
        self.rewards.flat_orientation_l2.weight = -3.0


@configclass
class SpiderResidualFlatEnvCfg_PLAY(SpiderResidualFlatEnvCfg):
    """Play/evaluation config on flat plane."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # forward-only at the speed cap, to check they actually sprint at 0.25 m/s
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
