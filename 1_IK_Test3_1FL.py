# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive IK target test for custom Spider Robot FL leg."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FL IK target test for custom Spider Robot.")
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
# ถ้า side view เพี้ยน ให้ลองเปลี่ยนเป็น:
# KNEE_MODE = "up"

LEG_SIGN = {
    "FL": {"coxa": -1, "femur": +1, "tibia": -1},
    "FR": {"coxa": -1, "femur": -1, "tibia": -1},
    "RL": {"coxa": -1, "femur": -1, "tibia": -1},
    "RR": {"coxa": -1, "femur": +1, "tibia": +1},
}

HIP_POS_BODY = {
    "FL": (-50.0, -35.0, 5.0),
}

STANDING_FOOT_BODY = {
    "FL": (-160.0, -115.0, -76.34),
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

    # if D > max_reach or D < min_reach:
    #     raise ValueError(
    #         f"Target unreachable: D={D:.2f} mm, "
    #         f"reach=[{min_reach:.2f}, {max_reach:.2f}] mm, "
    #         f"local target=({x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f})"
    #     )
    REACH_MARGIN = 5.0  # mm
    clamped = False

    safe_min = min_reach + REACH_MARGIN
    safe_max = max_reach - REACH_MARGIN

    if D > safe_max:
        scale = safe_max / D
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
    


    ### -------------- JUNK -------------------------

    # safe_min = min_reach + 5.0
    # safe_max = max_reach - 5.0
    # if D > safe_max:
    #     # reach_clamped = True
    #     # if reach_clamped:
    #     #     print("เกินขอบเขตแล้วโว้ยยยยยยบยยยยยย")
    #     # scale = max_reach / D

    #     # r *= scale
    #     # z_mm *= scale

    #     # D = max_reach
    #     scale = safe_max / D
    #     r *= scale
    #     z_mm *= scale
    #     D = safe_max
    #     clamped = True

    # if D < safe_min:
    #     scale = safe_min / max(D, 1e-6)
    #     r *= scale
    #     z_mm *= scale
    #     D = safe_min
    #     clamped = True
    
    #     # scale = min_reach / D

    #     # r *= scale
    #     # z_mm *= scale
    #     # D = min_reach
    ### ------------- JUNK -------------------
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

    return theta1, theta2, theta3


def body_ik_math(leg: str, foot_body_mm):
    foot_leg = body_to_leg_frame(leg, foot_body_mm)
    foot_local = leg_frame_to_ik_plane(leg, foot_leg)
    return raw_ik_math(*foot_local, knee=KNEE_MODE)


NEUTRAL_MATH = {
    "FL": body_ik_math("FL", STANDING_FOOT_BODY["FL"]),
}


def spider_ik_body(leg: str, x_body_mm: float, y_body_mm: float, z_body_mm: float):
    q_math = body_ik_math(leg, (x_body_mm, y_body_mm, z_body_mm))
    q_neutral = NEUTRAL_MATH[leg]

    sign = LEG_SIGN[leg]

    q_cmd = [
        sign["coxa"] * (q_math[0] - q_neutral[0]),
        sign["femur"] * (q_math[1] - q_neutral[1]),
        sign["tibia"] * (q_math[2] - q_neutral[2]),
    ]

    return q_cmd


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
        size=(0.025, 0.025, 0.025),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )

    root_z_guess_m = 0.120
    standing_fl = STANDING_FOOT_BODY["FL"]

    target_translation_m = (
        standing_fl[0] / 1000.0,
        standing_fl[1] / 1000.0,
        root_z_guess_m + standing_fl[2] / 1000.0,
    )

    target_cfg.func(
        "/World/IK_Target_FL",
        target_cfg,
        translation=target_translation_m,
    )

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
    print("[INFO]: Move /World/IK_Target_FL in the viewport to test FL IK.")

    default_root_state = spider.data.default_root_state.clone()
    default_joint_pos = spider.data.default_joint_pos.clone()
    default_joint_vel = spider.data.default_joint_vel.clone()

    coxa_id = spider.find_joints("coxa_FL_joint")[0][0]
    femur_id = spider.find_joints("femur_FL_joint")[0][0]
    tibia_id = spider.find_joints("tibia_FL_joint")[0][0]

    COXA_MIN, COXA_MAX = -0.70, 0.70
    FEMUR_MIN, FEMUR_MAX = -1.10, 1.10
    TIBIA_MIN, TIBIA_MAX = -1.20, 1.20

    count = 0

    q_smooth = default_joint_pos.clone()
    alpha = 0.08
    while simulation_app.is_running():

        if count % 1000 == 0:
            spider.write_root_pose_to_sim(default_root_state[:, :7])
            spider.write_root_velocity_to_sim(default_root_state[:, 7:])
            spider.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            spider.reset()
            print("\n[INFO]: Robot reset to cfg init_state.")

        joint_target = default_joint_pos.clone()

        try:
            target_world_m = get_prim_world_pos_m("/World/IK_Target_FL")
            root_world_m = spider.data.root_pos_w[0].detach().cpu().numpy()

            # World frame -> body frame
            # ตอนนี้สมมติว่า base ไม่ได้ rotate
            target_body_m = target_world_m - root_world_m
            target_body_mm = target_body_m * 1000.0

            q = spider_ik_body(
                "FL",
                float(target_body_mm[0]),
                float(target_body_mm[1]),
                float(target_body_mm[2]),
            )

            q[0] = clamp_joint(q[0], COXA_MIN, COXA_MAX)
            q[1] = clamp_joint(q[1], FEMUR_MIN, FEMUR_MAX)
            q[2] = clamp_joint(q[2], TIBIA_MIN, TIBIA_MAX)

            

            joint_target[:, coxa_id] = (1 - alpha) * q_smooth[:, coxa_id] + alpha * q[0]
            joint_target[:, femur_id] = (1 - alpha) * q_smooth[:, femur_id] + alpha * q[1]
            joint_target[:, tibia_id] = (1 - alpha) * q_smooth[:, tibia_id] + alpha * q[2]

            q_smooth = joint_target.clone()

            # joint_target[:, coxa_id] = q[0]
            # joint_target[:, femur_id] = q[1]
            # joint_target[:, tibia_id] = q[2]

            if count % 100 == 0:
                print(
                    f"[IK FL] target_body_mm=({target_body_mm[0]:.1f}, "
                    f"{target_body_mm[1]:.1f}, {target_body_mm[2]:.1f}) "
                    f"q_deg=({math.degrees(q[0]):.2f}, "
                    f"{math.degrees(q[1]):.2f}, {math.degrees(q[2]):.2f})"
                )

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