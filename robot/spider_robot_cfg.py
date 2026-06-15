# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom 4-leg spider robot.

Robot structure from exported URDF:
- base_link
- Coxa_FL_1, Femur_FL_1, Tibia_FL_1
- Coxa_FR_1, Femur_FR_1, Tibia_FR_1
- Coxa_RL_1, Femur_RL_1, Tibia_RL_1
- Coxa_RR_1, Femur_RR_1, Tibia_RR_1

Joint naming:
- coxa_FL_joint,  coxa_FR_joint,  coxa_RL_joint,  coxa_RR_joint
- femur_FL_joint, femur_FR_joint, femur_RL_joint, femur_RR_joint
- tibia_FL_joint, tibia_FR_joint, tibia_RL_joint, tibia_RR_joint

Use this file like Isaac Lab's unitree.py robot config file.
SPIDER_USD_PATH is resolved relative to this file, so no per-machine edit is needed
as long as the Robot_USD_v.1.02Final/ folder stays next to this config.

Actuator: ImplicitActuatorCfg (PhysX internal drive, unconditionally stable). The old
explicit DCMotorCfg (archived in spider_robot_cfg_OLD_DCMotor_unused.py) couldn't be
damped enough on these very light links, so the joints whipped on every swing.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import math
import os


# USD path, resolved relative to this file so the project is portable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_USD_PATH = os.path.join(
    _THIS_DIR, "Robot_USD_v.1.02Final", "SpiderBot_v1.02_Final.usd"
)


# -----------------------------------------------------------------------------
# Joint groups
# -----------------------------------------------------------------------------
COXA_JOINTS = [
    "coxa_FL_joint",
    "coxa_FR_joint",
    "coxa_RL_joint",
    "coxa_RR_joint",
]

FEMUR_JOINTS = [
    "femur_FL_joint",
    "femur_FR_joint",
    "femur_RL_joint",
    "femur_RR_joint",
]

TIBIA_JOINTS = [
    "tibia_FL_joint",
    "tibia_FR_joint",
    "tibia_RL_joint",
    "tibia_RR_joint",
]

LEG_JOINTS = COXA_JOINTS + FEMUR_JOINTS + TIBIA_JOINTS

FOOT_BODY_NAMES = [
    "Foot_FL_1",
    "Foot_FR_1",
    "Foot_RL_1",
    "Foot_RR_1",
]


# -----------------------------------------------------------------------------
# Articulation config
# -----------------------------------------------------------------------------
SPIDER_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SPIDER_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # standing height ~105 mm; spawn slightly higher for ground clearance
        pos=(0.0, 0.0, 0.120),
        joint_pos={
            ".*coxa_.*_joint": math.radians(0),
            ".*femur_.*_joint": math.radians(0),
            "tibia_FL_joint": math.radians(0),
            "tibia_RR_joint": math.radians(0),
            "tibia_FR_joint": math.radians(0),
            "tibia_RL_joint": math.radians(0),
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # implicit PD actuator (see the actuator note in the module docstring)
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*coxa_.*_joint", ".*femur_.*_joint", ".*tibia_.*_joint"],
            effort_limit=100.0,
            velocity_limit=45.0,
            stiffness={
                ".*coxa_.*_joint": 10.0,
                ".*femur_.*_joint": 30.0,
                ".*tibia_.*_joint": 30.0,
            },
            # well-damped (~ζ0.7), now possible under the stable implicit drive
            damping={
                ".*coxa_.*_joint": 0.5,
                ".*femur_.*_joint": 0.3,
                ".*tibia_.*_joint": 0.2,
            },
            # armature: reflected rotor inertia; damps jitter on the very light links
            armature={
                ".*coxa_.*_joint": 0.001,
                ".*femur_.*_joint": 0.001,
                ".*tibia_.*_joint": 0.001,
            },
            friction=0.05,
        ),
    },
)
"""Custom spider robot config using an implicit PD actuator model."""


# Common alias if you prefer shorter import names.
SPIDER_CFG = SPIDER_ROBOT_CFG
