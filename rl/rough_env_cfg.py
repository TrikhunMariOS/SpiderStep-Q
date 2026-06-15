# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain residual-gait velocity task for the Spider robot.

A standard Go2-style LocomotionVelocityRoughEnvCfg with two key changes:
  1. ACTION: GaitFootOffsetAction — RL nudges each foot target on top of the
     procedural gait instead of learning to walk from scratch.
  2. OBS:    + gait_phase (sin/cos) so the residual knows where in the step cycle it is.
"""

import math
import os
import sys

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

# add the project root (MY_Final) to path so `robot` resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.spider_robot_cfg import SPIDER_ROBOT_CFG, FOOT_BODY_NAMES  # noqa: E402

from . import mdp  # noqa: E402


BASE_BODY_NAME = "base_link"
FOOT_BODY_NAME_EXPR = "Foot_.*_1"


@configclass
class SpiderResidualRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Residual-gait training environment on rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 30.0

        # scene / robot
        self.scene.robot = SPIDER_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Spider_Robot_v1_02_ex_urdf/base_link"
        # finer scanner over a smaller patch: 0.03m cells, 0.5x0.4m (~238 rays) so box edges
        # don't fall in the gaps between rays
        self.scene.height_scanner.pattern_cfg.resolution = 0.03
        self.scene.height_scanner.pattern_cfg.size = (0.5, 0.4)

        # the USD bodies don't register a contact-reporter API, so the contact sensor can't
        # init. Residual RL doesn't need foot-contact rewards; re-export the USD to re-enable.
        self.scene.contact_forces = None

        self.sim.dt = 0.0025              # robot/gait tuned at 0.0025; do not raise
        self.decimation = 8               # policy step = 0.0025 × 8 = 0.02 s (50 Hz)
        self.scene.num_envs = 512
        self.scene.env_spacing = 2.0

        # ACTION: Cartesian foot-offset on top of the gait. Δz-only is the safe start
        # (offset_mask=(0,0,1)); offset_scale_mm=0 must walk exactly like the baseline.
        self.actions.joint_pos = mdp.GaitFootOffsetActionCfg(
            asset_name="robot",
            # smaller residual = closer to the proven gait; action_l2 makes it self-scale
            # (~0 on flat, larger on obstacles). Δz big for clearance, Δxy small.
            offset_scale_mm=45.0,
            offset_mask=(0.4, 0.4, 1.0),  # ±18mm placement / ±45mm height
            # dynamic-freq channel frozen (policy never used it); run a fixed 1.4Hz cadence
            # -> 1.4Hz × 100mm stride ≈ 0.14 m/s
            step_frequency=1.4,
            freq_base=1.4,
            freq_up=0.0,
            freq_down=0.0,
            enable_cog_shift=True,        # keep test5 feedforward CoG as baseline
            command_name="base_velocity",
            settle_steps=120,             # 0.3 s settle before walking
            command_signs=(1.0, 1.0, 1.0),
        )

        # OBS: gait phase (sin/cos) so the residual is phase-aware
        self.observations.policy.gait_phase = ObsTerm(func=mdp.gait_phase)

        # commands: speed cap at fixed 1.4Hz ≈ 0.14 m/s. vy/turn a bit lower since the
        # fore-aft leg layout makes strafe the naturally weaker mode.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.14, 0.14)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.12, 0.12)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.rel_heading_envs = 0.0
        # long holds so the robot nets enough distance to promote in the curriculum,
        # but still ~2-3 commands/episode so transitions are practiced
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)

        # terrain scaled for this small (~10cm) robot
        if self.scene.terrain.terrain_generator is not None:
            gen = self.scene.terrain.terrain_generator
            # 2m tiles: the curriculum promotes after walking past size/2, so a small tile
            # keeps the promote bar (1.0m) reachable for this slow crawler
            gen.size = (2.0, 2.0)
            terrains = gen.sub_terrains
            if "boxes" in terrains:
                terrains["boxes"].grid_height_range = (0.015, 0.050)
            if "random_rough" in terrains:
                terrains["random_rough"].noise_range = (0.004, 0.025)
                terrains["random_rough"].noise_step = 0.005

            # 20% of every level is dead-flat so the robot keeps practicing flat walking.
            # noise_step must be >= vertical_scale (0.005) or arange divides by zero.
            terrains["flat"] = terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2, noise_range=(0.0, 0.0), noise_step=0.005, border_width=0.25,
            )

            # stairs rescaled to this robot: 20-45mm steps, narrow treads, 1m platform
            for key in ("pyramid_stairs", "pyramid_stairs_inv"):
                if key in terrains:
                    terrains[key].proportion = 0.1
                    terrains[key].step_height_range = (0.02, 0.045)
                    terrains[key].step_width = 0.25
                    terrains[key].platform_width = 1.0
                    terrains[key].border_width = 0.25

            # gaps 2-8cm. ~8cm is the physical ceiling for a static step (stride+Δx reach);
            # wider needs a dynamic leap this crawl gait can't do.
            terrains["gap"] = terrain_gen.MeshGapTerrainCfg(
                proportion=0.15,
                gap_width_range=(0.02, 0.08),
                platform_width=1.0,
            )

        # curriculum stays ON (default easy->hard). Training on uniformly-hard terrain from
        # step 0 produced a braced gait that walked worse even on flat.

        # events / domain randomization. Ranges rescaled for this tiny robot (base ≈ 1.0 kg);
        # the stock ranges are sized for a 50 kg quadruped.
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)

        # base mass: payload/build tolerance, ±range keeps it 0.8-1.4 kg
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
                "mass_distribution_params": (-0.2, 0.4),
                "operation": "add",
            },
        )

        # push recovery: ±1.0 m/s (≈8× walk speed). ±3.0 was too brutal and slowed learning.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 7.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )

        # add continuous force/torque + COM shift in a later DR round
        self.events.base_external_force_torque = None
        self.events.base_com = None

        # spawn: random heading (base-frame command), zero velocity (push is the disturbance)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time = None   # dropped with the contact sensor

        self.rewards.flat_orientation_l2.weight = -7.0
        # sharp tracking std + strong ang weight: the fix for "strafe -> body yaws" (the
        # crawl's diagonal lift order induces a yaw moment the policy must cancel)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.30
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.1        # anti-jitter (underdamped joints)

        # residual-magnitude penalty: keeps RL close to the gait and self-scaling
        self.rewards.action_l2 = self.rewards.action_rate_l2.__class__(
            func=mdp.action_l2, weight=-0.05,
        )

        # anti-degenerate-gait: penalize pushing a foot down during its swing (the policy
        # was suppressing rear-leg lift to dodge roll, leaving feet dragging)
        self.rewards.foot_swing_clearance = RewTerm(
            func=mdp.foot_swing_suppression,
            weight=-1.0,
            params={"action_term_name": "joint_pos"},
        )

        # perception: swing-foot clearance over the terrain under each foot. On a box the
        # local ground is higher, so scoring this requires reading height_scan.
        self.rewards.foot_clearance_terrain = RewTerm(
            func=mdp.foot_clearance_over_terrain,
            weight=-2.0,
            params={
                "action_term_name": "joint_pos",
                "sensor_name": "height_scanner",
                "target_clearance": 0.04,   # 40mm above local ground at swing peak
                "foot_radius": 0.010,       # FK tracks ball center; contact is 10mm lower
                "neighborhood_radius": 0.06,  # clear the highest ground within 6cm of the foot
            },
        )

        # foot_touchdown_impact was tried and reverted: braking the descent delays touchdown
        # but the phase-locked CoG shift then tips the body. Needs touchdown-aware timing.

        self.rewards.undesired_contacts = None

        # terminations: contact-free, terminate when tipped past 50°
        self.terminations.base_contact = None
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": math.radians(50.0)}
        )


@configclass
class SpiderResidualRoughEnvCfg_PLAY(SpiderResidualRoughEnvCfg):
    """Play/evaluation config."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # draw the height-scanner ray hits in the viewport (play only — costs FPS)
        self.scene.height_scanner.debug_vis = True

        # forward-only at the cadence cap (0.14 m/s)
        self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.14)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
