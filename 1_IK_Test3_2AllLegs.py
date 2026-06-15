# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive IK target test for custom Spider Robot 4 legs.

Usage:
    isaaclab.bat -p IK_Test4_AllLegs.py

This script spawns 4 target cubes:
    /World/IK_Target_FL
    /World/IK_Target_FR
    /World/IK_Target_RL
    /World/IK_Target_RR
Move each cube slowly in the viewport to test each leg IK.

Notes:
    - IK constants use millimeters.
    - Isaac Lab positions are meters, so this script converts meter -> mm internally.
    - Joint commands are radians.
    - This assumes robot base/root is not rotating during the test.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="4-leg IK target test for custom Spider Robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from pxr import Usd, UsdGeom
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from robot.spider_robot_cfg import SPIDER_CFG


# =============================================================================
# IK MODEL PARAMETERS, unit = mm
# =============================================================================

L1 = 35.0
L2 = 65.0
L3 = 100.0

KNEE_MODE = "down"

# IMPORTANT:
# FL is confirmed by your latest test:
#     FL = coxa -1, femur +1, tibia -1
#
# FR/RL/RR are starting guesses based on your current file and mirror pattern.
# Test each leg and adjust one sign at a time if a leg moves in the wrong direction.
LEG_SIGN = {
    "FL": {"coxa": -1, "femur": +1, "tibia": -1},
    "FR": {"coxa": -1, "femur": -1, "tibia": +1},
    "RL": {"coxa": -1, "femur": -1, "tibia": +1},
    "RR": {"coxa": -1, "femur": +1, "tibia": -1},
}

HIP_POS_BODY = {
    "FL": (-50.0, -35.0, 5.0),
    "FR": (-50.0,  35.0, 5.0),
    "RL": ( 50.0, -35.0, 5.0),
    "RR": ( 50.0,  35.0, 5.0),
}

STANDING_FOOT_BODY = {
    "FL": (-161.0, -113.0, -76.3),
    "FR": (-161.0,  113.0, -76.3),
    "RL": ( 161.0, -113.0, -76.3),
    "RR": ( 161.0,  113.0, -76.3),
}

LEG_JOINTS = {
    "FL": {"coxa": "coxa_FL_joint", "femur": "femur_FL_joint", "tibia": "tibia_FL_joint"},
    "FR": {"coxa": "coxa_FR_joint", "femur": "femur_FR_joint", "tibia": "tibia_FR_joint"},
    "RL": {"coxa": "coxa_RL_joint", "femur": "femur_RL_joint", "tibia": "tibia_RL_joint"},
    "RR": {"coxa": "coxa_RR_joint", "femur": "femur_RR_joint", "tibia": "tibia_RR_joint"},
}

TARGET_PRIMS = {
    "FL": "/World/IK_Target_FL",
    "FR": "/World/IK_Target_FR",
    "RL": "/World/IK_Target_RL",
    "RR": "/World/IK_Target_RR",
}

JOINT_LIMITS = {
    "coxa":  (-0.70, 0.70),
    "femur": (-1.10, 1.10),
    "tibia": (-1.20, 1.20),
}


# =============================================================================
# IK FUNCTIONS
# =============================================================================

def body_to_leg_frame(leg: str, foot_body_mm):
    fx, fy, fz = foot_body_mm
    hx, hy, hz = HIP_POS_BODY[leg]
    return fx - hx, fy - hy, fz - hz


def leg_frame_to_ik_plane(leg: str, foot_leg_mm):
    x, y, z = foot_leg_mm

    sx, sy, _ = body_to_leg_frame(leg, STANDING_FOOT_BODY[leg])
    base_angle = math.atan2(sy, sx)

    c = math.cos(-base_angle)
    s = math.sin(-base_angle)

    x_local = c * x - s * y
    y_local = s * x + c * y

    return x_local, y_local, z


def raw_ik_math(x_mm: float, y_mm: float, z_mm: float, knee: str = "down"):
    theta1 = math.atan2(y_mm, x_mm)

    horizontal = math.sqrt(x_mm * x_mm + y_mm * y_mm)
    r = horizontal - L1
    D = math.sqrt(r * r + z_mm * z_mm)

    min_reach = abs(L2 - L3)
    max_reach = L2 + L3

    REACH_MARGIN = 5.0
    safe_min = min_reach + REACH_MARGIN
    safe_max = max_reach - REACH_MARGIN

    clamped = False

    if D > safe_max:
        scale = safe_max / max(D, 1e-6)
        r *= scale
        z_mm *= scale
        D = safe_max
        clamped = True

    elif D < safe_min:
        scale = safe_min / max(D, 1e-6)
        r *= scale
        z_mm *= scale
        D = safe_min
        clamped = True

    cos_alpha = (L2**2 + D**2 - L3**2) / (2 * L2 * D)
    cos_beta = (L2**2 + L3**2 - D**2) / (2 * L2 * L3)

    alpha = math.acos(np.clip(cos_alpha, -1.0, 1.0))
    beta = math.acos(np.clip(cos_beta, -1.0, 1.0))

    base = math.atan2(z_mm, r)

    if knee == "down":
        theta2 = base + alpha
        theta3 = math.pi - beta
    elif knee == "up":
        theta2 = base - alpha
        theta3 = -(math.pi - beta)
    else:
        raise ValueError(f"Unknown knee mode: {knee}")

    return theta1, theta2, theta3, clamped


def body_ik_math(leg: str, foot_body_mm):
    foot_leg = body_to_leg_frame(leg, foot_body_mm)
    foot_local = leg_frame_to_ik_plane(leg, foot_leg)
    return raw_ik_math(*foot_local, knee=KNEE_MODE)


NEUTRAL_MATH = {}
for leg in ("FL", "FR", "RL", "RR"):
    q0, q1, q2, _ = body_ik_math(leg, STANDING_FOOT_BODY[leg])
    NEUTRAL_MATH[leg] = (q0, q1, q2)


def spider_ik_body(leg: str, x_body_mm: float, y_body_mm: float, z_body_mm: float):
    q0, q1, q2, clamped = body_ik_math(leg, (x_body_mm, y_body_mm, z_body_mm))
    q_math = (q0, q1, q2)
    q_neutral = NEUTRAL_MATH[leg]
    sign = LEG_SIGN[leg]

    q_cmd = [
        sign["coxa"]  * (q_math[0] - q_neutral[0]),
        sign["femur"] * (q_math[1] - q_neutral[1]),
        sign["tibia"] * (q_math[2] - q_neutral[2]),
    ]

    return q_cmd, clamped


def get_prim_world_pos_m(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pos = mat.ExtractTranslation()

    return np.array([pos[0], pos[1], pos[2]], dtype=np.float64)


def clamp_joint(q, min_rad, max_rad):
    return max(min(q, max_rad), min_rad)


# =============================================================================
# SCENE
# =============================================================================

def spawn_target_cube(leg: str, color):
    target_cfg = sim_utils.MeshCuboidCfg(
        size=(0.025, 0.025, 0.025),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )

    root_z_guess_m = 0.120
    foot = STANDING_FOOT_BODY[leg]

    target_translation_m = (
        foot[0] / 1000.0,
        foot[1] / 1000.0,
        root_z_guess_m + foot[2] / 1000.0,
    )

    target_cfg.func(TARGET_PRIMS[leg], target_cfg, translation=target_translation_m)


def design_scene() -> Articulation:
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/lightDistant", light_cfg, translation=(1.0, 0.0, 10.0))

    spider_cfg = SPIDER_CFG.replace(prim_path="/World/Spider")
    spider = Articulation(spider_cfg)

    spawn_target_cube("FL", (0.0, 1.0, 0.0))
    spawn_target_cube("FR", (0.0, 0.5, 1.0))
    spawn_target_cube("RL", (1.0, 0.5, 0.0))
    spawn_target_cube("RR", (1.0, 0.0, 1.0))

    return spider


# =============================================================================
# MAIN
# =============================================================================

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[0.8, 0.8, 0.50],
        target=[0.0, 0.0, 0.08],
    )

    spider = design_scene()
    sim.reset()

    print("[INFO]: Setup complete.")
    print(f"[INFO]: Joint names: {spider.joint_names}")
    print(f"[INFO]: Body names: {spider.body_names}")
    print(f"[INFO]: KNEE_MODE = {KNEE_MODE}")
    print("[INFO]: Move target cubes in the viewport to test IK.")
    for leg, path in TARGET_PRIMS.items():
        print(f"  {leg}: {path}")

    default_root_state = spider.data.default_root_state.clone()
    default_joint_pos = spider.data.default_joint_pos.clone()
    default_joint_vel = spider.data.default_joint_vel.clone()

    joint_ids = {}
    for leg, names in LEG_JOINTS.items():
        joint_ids[leg] = {
            "coxa": spider.find_joints(names["coxa"])[0][0],
            "femur": spider.find_joints(names["femur"])[0][0],
            "tibia": spider.find_joints(names["tibia"])[0][0],
        }

    q_smooth = default_joint_pos.clone()
    alpha = 0.08

    count = 0

    while simulation_app.is_running():

        if count % 1000 == 0:
            spider.write_root_pose_to_sim(default_root_state[:, :7])
            spider.write_root_velocity_to_sim(default_root_state[:, 7:])
            spider.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            spider.reset()
            q_smooth = default_joint_pos.clone()
            print("\\n[INFO]: Robot reset to cfg init_state.")

        joint_target = default_joint_pos.clone()

        try:
            root_world_m = spider.data.root_pos_w[0].detach().cpu().numpy()
            debug_chunks = []

            for leg in ("FL", "FR", "RL", "RR"):
                target_world_m = get_prim_world_pos_m(TARGET_PRIMS[leg])

                # World frame -> body frame.
                # Assumption: base/root is not rotated during this IK test.
                target_body_m = target_world_m - root_world_m
                target_body_mm = target_body_m * 1000.0

                q, clamped = spider_ik_body(
                    leg,
                    float(target_body_mm[0]),
                    float(target_body_mm[1]),
                    float(target_body_mm[2]),
                )

                q[0] = clamp_joint(q[0], *JOINT_LIMITS["coxa"])
                q[1] = clamp_joint(q[1], *JOINT_LIMITS["femur"])
                q[2] = clamp_joint(q[2], *JOINT_LIMITS["tibia"] )

                ids = joint_ids[leg]

                joint_target[:, ids["coxa"]] = (1.0 - alpha) * q_smooth[:, ids["coxa"]] + alpha * q[0]
                joint_target[:, ids["femur"]] = (1.0 - alpha) * q_smooth[:, ids["femur"]] + alpha * q[1]
                joint_target[:, ids["tibia"]] = (1.0 - alpha) * q_smooth[:, ids["tibia"]] + alpha * q[2]

                if count % 100 == 0:
                    mark = " CLAMP" if clamped else ""
                    debug_chunks.append(
                        f"{leg}{mark}: target=({target_body_mm[0]:.0f},{target_body_mm[1]:.0f},{target_body_mm[2]:.0f}) "
                        f"q=({math.degrees(q[0]):.1f},{math.degrees(q[1]):.1f},{math.degrees(q[2]):.1f})"
                    )

            q_smooth = joint_target.clone()

            if count % 100 == 0:
                print("[IK] " + " | ".join(debug_chunks))

        except Exception as e:
            if count % 100 == 0:
                print(f"[IK ERROR] {e}")

        spider.set_joint_position_target(joint_target)

        spider.write_data_to_sim()
        sim.step()
        spider.update(sim.get_physics_dt())

        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
