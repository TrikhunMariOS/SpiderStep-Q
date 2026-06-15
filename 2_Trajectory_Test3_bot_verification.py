# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""FL trajectory test for custom Spider Robot.

This script:
    - Spawns your spider robot.
    - Spawns one green target cube for FL.
    - Moves the target cube along an ellipse-like trajectory automatically.
    - Runs IK every frame.
    - Commands only FL joints to follow the trajectory.

Usage:
    isaaclab.bat -p IK_Trajectory_FL.py
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FL trajectory IK test for custom Spider Robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from pxr import UsdGeom, Gf
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

LEG_SIGN = {
    "FL": {"coxa": -1, "femur": +1, "tibia": -1},
}

HIP_POS_BODY = {
    "FL": (-50.0, -35.0, 5.0),
}

STANDING_FOOT_CONTACT_BODY = {
    "FL": (-160.0, -115.0, -76.34),
}

FOOT_RADIUS = 10.0

STANDING_FOOT_BODY = {
    leg: (x, y, z + FOOT_RADIUS)
    for leg, (x, y, z) in STANDING_FOOT_CONTACT_BODY.items()
}

# =============================================================================
# TRAJECTORY PARAMETERS, unit = mm
# =============================================================================

STEP_FREQUENCY = 1.0      # Hz
STRIDE_LENGTH = 50.0       # mm, start small
STEP_HEIGHT = 30.0         # mm, start small
FORWARD_SIGN_X = -1.0      # robot front direction is roughly -X


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

    # Keep targets away from exact workspace boundary.
    reach_margin = 5.0
    safe_min = min_reach + reach_margin
    safe_max = max_reach - reach_margin

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

def fk_math(x0_body, y0_body, z0_body, q_cmd):
    """Approx FK for visualization only.

    Input:
        q_cmd = robot joint command [coxa, femur, tibia]
    Output:
        foot position in body frame, mm
    """

    sign = LEG_SIGN["FL"]

    # robot command -> math angle
    q_neutral = NEUTRAL_MATH["FL"]

    theta1 = q_neutral[0] + sign["coxa"] * q_cmd[0]
    theta2 = q_neutral[1] + sign["femur"] * q_cmd[1]
    theta3 = q_neutral[2] + sign["tibia"] * q_cmd[2]

    # FK in IK local plane
    r = L1 + L2 * math.cos(theta2) + L3 * math.cos(theta2 - theta3)
    z = L2 * math.sin(theta2) + L3 * math.sin(theta2 - theta3)

    x_local = r * math.cos(theta1)
    y_local = r * math.sin(theta1)

    # local IK plane -> leg frame
    sx, sy, _ = body_to_leg_frame("FL", STANDING_FOOT_BODY["FL"])
    base_angle = math.atan2(sy, sx)

    c = math.cos(base_angle)
    s = math.sin(base_angle)

    x_leg = c * x_local - s * y_local
    y_leg = s * x_local + c * y_local
    z_leg = z

    # leg frame -> body frame
    hx, hy, hz = HIP_POS_BODY["FL"]

    return (
        hx + x_leg,
        hy + y_leg,
        hz + z_leg,
    )

def body_ik_math(leg: str, foot_body_mm):
    foot_leg = body_to_leg_frame(leg, foot_body_mm)
    foot_local = leg_frame_to_ik_plane(leg, foot_leg)
    return raw_ik_math(*foot_local, knee=KNEE_MODE)


NEUTRAL_MATH_FULL = {
    "FL": body_ik_math("FL", STANDING_FOOT_BODY["FL"]),
}
NEUTRAL_MATH = {
    leg: angles[:3] for leg, angles in NEUTRAL_MATH_FULL.items()
}


def spider_ik_body(leg: str, x_body_mm: float, y_body_mm: float, z_body_mm: float):
    q0, q1, q2, clamped = body_ik_math(leg, (x_body_mm, y_body_mm, z_body_mm))
    q_math = (q0, q1, q2)
    q_neutral = NEUTRAL_MATH[leg]
    sign = LEG_SIGN[leg]

    q_cmd = [
        sign["coxa"] * (q_math[0] - q_neutral[0]),
        sign["femur"] * (q_math[1] - q_neutral[1]),
        sign["tibia"] * (q_math[2] - q_neutral[2]),
    ]

    return q_cmd, clamped


def clamp_joint(q, min_rad, max_rad):
    return max(min(q, max_rad), min_rad)


# =============================================================================
# TRAJECTORY
# =============================================================================

def fl_ellipse_trajectory_body(t: float):
    """Generate FL foot target in body frame.

    phase 0.0 -> 0.5:
        swing phase, foot moves forward and lifts up.

    phase 0.5 -> 1.0:
        stance phase, foot returns backward near ground.
    """
    x0, y0, z0 = STANDING_FOOT_BODY["FL"]
    phase = (t * STEP_FREQUENCY) % 1.0

    if phase < 0.5:
        u = phase / 0.5
        x = x0 - FORWARD_SIGN_X * STRIDE_LENGTH / 2.0 + FORWARD_SIGN_X * STRIDE_LENGTH * u
        z = z0 + STEP_HEIGHT * math.sin(math.pi * u)
    else:
        u = (phase - 0.5) / 0.5
        x = x0 + FORWARD_SIGN_X * STRIDE_LENGTH / 2.0 - FORWARD_SIGN_X * STRIDE_LENGTH * u
        z = z0

    y = y0
    return (x, y, z), phase


def set_prim_world_pos_m(prim_path: str, pos_m):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    translate_ops = [
        op for op in xform.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]

    vec = Gf.Vec3d(float(pos_m[0]), float(pos_m[1]), float(pos_m[2]))
    if translate_ops:
        translate_ops[0].Set(vec)
    else:
        xform.AddTranslateOp().Set(vec)


# =============================================================================
# SCENE
# =============================================================================

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

    target_cfg = sim_utils.MeshCuboidCfg(
        size=(0.005, 0.005, 0.005),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )

    fk_cfg = sim_utils.MeshCuboidCfg(
        size=(0.005, 0.005, 0.005),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )

    actual_cfg = sim_utils.MeshCuboidCfg(
        size=(0.010, 0.010, 0.010),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 1.0)),
    )

    root_z_guess_m = 0.120
    standing_fl = STANDING_FOOT_BODY["FL"]
    target_translation_m = (
        standing_fl[0] / 1000.0,
        standing_fl[1] / 1000.0,
        root_z_guess_m + standing_fl[2] / 1000.0,
    )

    target_cfg.func("/World/IK_Target_FL", target_cfg, translation=target_translation_m)
    fk_cfg.func("/World/FK_Result_FL", fk_cfg, translation=target_translation_m)
    actual_cfg.func("/World/Actual_Foot_FL", actual_cfg, translation=target_translation_m)
    return spider


# =============================================================================
# MAIN
# =============================================================================

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[0.7, 0.7, 0.45],
        target=[0.0, 0.0, 0.08],
    )

    spider = design_scene()
    sim.reset()

    print("[INFO]: Setup complete.")
    print(f"[INFO]: Joint names: {spider.joint_names}")
    print(f"[INFO]: Body names: {spider.body_names}")
    print(f"[INFO]: KNEE_MODE = {KNEE_MODE}")
    print("[INFO]: FL target cube will move along an ellipse-like trajectory.")

    default_root_state = spider.data.default_root_state.clone()
    default_joint_pos = spider.data.default_joint_pos.clone()
    default_joint_vel = spider.data.default_joint_vel.clone()

    coxa_id = spider.find_joints("coxa_FL_joint")[0][0]
    femur_id = spider.find_joints("femur_FL_joint")[0][0]
    tibia_id = spider.find_joints("tibia_FL_joint")[0][0]

    # Actual body position from Isaac/PhysX.
    # NOTE: This is the body origin of Foot_FL_1, not necessarily the exact foot contact point.
    foot_body_id = spider.find_bodies("Foot_FL_1")[0][0]

    COXA_MIN, COXA_MAX = -0.70, 0.70
    FEMUR_MIN, FEMUR_MAX = -1.10, 1.10
    TIBIA_MIN, TIBIA_MAX = -1.20, 1.20

    q_smooth = default_joint_pos.clone()
    alpha = 0.08
    count = 0

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        t = count * dt

        if count % 2000 == 0:
            spider.write_root_pose_to_sim(default_root_state[:, :7])
            spider.write_root_velocity_to_sim(default_root_state[:, 7:])
            spider.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            spider.reset()
            q_smooth = default_joint_pos.clone()
            print("\n[INFO]: Robot reset to cfg init_state.")

        joint_target = default_joint_pos.clone()

        try:
            root_world_m = spider.data.root_pos_w[0].detach().cpu().numpy()
            foot_body_mm, phase = fl_ellipse_trajectory_body(t)

            # Move visual target cube along the same body-frame trajectory.
            target_world_m = root_world_m + np.array(foot_body_mm, dtype=np.float64) / 1000.0
            set_prim_world_pos_m("/World/IK_Target_FL", target_world_m)

            q, clamped = spider_ik_body("FL", *foot_body_mm)

            fk_body_mm = fk_math(
                STANDING_FOOT_BODY["FL"][0],
                STANDING_FOOT_BODY["FL"][1],
                STANDING_FOOT_BODY["FL"][2],
                q,
            )

            fk_world_m = root_world_m + np.array(fk_body_mm, dtype=np.float64) / 1000.0
            set_prim_world_pos_m("/World/FK_Result_FL", fk_world_m)

            # Actual simulated body position from Isaac.
            actual_foot_world_m = spider.data.body_pos_w[0, foot_body_id].detach().cpu().numpy()
            set_prim_world_pos_m("/World/Actual_Foot_FL", actual_foot_world_m)

            q[0] = clamp_joint(q[0], COXA_MIN, COXA_MAX)
            q[1] = clamp_joint(q[1], FEMUR_MIN, FEMUR_MAX)
            q[2] = clamp_joint(q[2], TIBIA_MIN, TIBIA_MAX)

            joint_target[:, coxa_id] = (1.0 - alpha) * q_smooth[:, coxa_id] + alpha * q[0]
            joint_target[:, femur_id] = (1.0 - alpha) * q_smooth[:, femur_id] + alpha * q[1]
            joint_target[:, tibia_id] = (1.0 - alpha) * q_smooth[:, tibia_id] + alpha * q[2]

            q_smooth = joint_target.clone()

            if count % 100 == 0:
                phase_name = "SWING" if phase < 0.5 else "STANCE"
                clamp_text = " CLAMPED" if clamped else ""
                print(
                    f"[TRAJ FL] phase={phase:.2f} {phase_name}{clamp_text} "
                    f"target_body_mm=({foot_body_mm[0]:.1f}, {foot_body_mm[1]:.1f}, {foot_body_mm[2]:.1f}) "
                    f"q_deg=({math.degrees(q[0]):.2f}, {math.degrees(q[1]):.2f}, {math.degrees(q[2]):.2f})"
                )
                fk_err = np.linalg.norm(np.array(fk_body_mm) - np.array(foot_body_mm))
                actual_err = np.linalg.norm((actual_foot_world_m - target_world_m) * 1000.0)
                model_vs_actual_err = np.linalg.norm((actual_foot_world_m - fk_world_m) * 1000.0)

                print(f"FK math error          = {fk_err:.2f} mm")
                print(f"Actual foot error      = {actual_err:.2f} mm")
                print(f"Model vs Isaac error   = {model_vs_actual_err:.2f} mm")

        except Exception as e:
            if count % 100 == 0:
                print(f"[IK ERROR] {e}")

        spider.set_joint_position_target(joint_target)
        spider.write_data_to_sim()
        sim.step()
        spider.update(dt)
        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
