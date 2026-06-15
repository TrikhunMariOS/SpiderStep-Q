# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Gait Test 2: Stable crawl gait for custom Spider Robot.

Improvements over Gait Test 1:
    - Shorter swing phase, longer stance phase.
    - Smaller stride for stability.
    - Body-shift compensation by shifting foot targets opposite to desired body sway.
    - Still procedural only, no RL yet.

Usage:
    isaaclab.bat -p Gait_Test2_Stable_Crawl.py
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stable crawl gait test for custom Spider Robot.")
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
# ROBOT / IK PARAMETERS, unit = mm
# =============================================================================

L1 = 35.0
L2 = 65.0
L3 = 100.0

FOOT_RADIUS = 10.0
KNEE_MODE = "down"

LEGS = ["FL", "FR", "RL", "RR"]

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

# Measured bottom contact point at neutral standing pose.
# STANDING_FOOT_CONTACT_BODY = {
#     "FL": (-150.0, -105.0, -56.34),
#     "FR": (-150.0,  105.0, -56.34),
#     "RL": ( 150.0, -105.0, -56.34),
#     "RR": ( 150.0,  105.0, -56.34),
# }
STANDING_FOOT_CONTACT_BODY = {
    "FL": (-160.0, -115.0, -76.34),
    "FR": (-160.0,  115.0, -76.34),
    "RL": ( 160.0, -115.0, -76.34),
    "RR": ( 160.0,  115.0, -76.34),
}
# IK/FK/trajectory use foot center, not bottom contact point.
STANDING_FOOT_BODY = {
    leg: (x, y, z + FOOT_RADIUS)
    for leg, (x, y, z) in STANDING_FOOT_CONTACT_BODY.items()
}


# =============================================================================
# GAIT PARAMETERS
# =============================================================================

STEP_FREQUENCY = 1      # Hz, slow first
STRIDE_LENGTH = 60.0       # mm, smaller = more stable
STEP_HEIGHT = 45.0         # mm
FORWARD_SIGN_X = -1.0      # robot front direction is approximately -X

# Stable crawl: each leg swings only 25% of cycle.
# More time on stance = more stable support.
SWING_RATIO = 0.25

# Crawl order: FL -> RR -> FR -> RL
PHASE_OFFSET = {
    "FL": 0.00,
    "RR": 0.25,
    "FR": 0.50,
    "RL": 0.75,
}

# Procedural body shift compensation.
# We do not move the real root body here.
# Instead, we offset all foot targets opposite to the desired body sway.
# This approximates shifting the body center toward the support triangle.
BODY_SHIFT_X = 5.0      # mm
BODY_SHIFT_Y = 16.0     # mm

# Joint target smoothing.
ALPHA = 0.20


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


def body_ik_math(leg: str, foot_body_mm):
    foot_leg = body_to_leg_frame(leg, foot_body_mm)
    foot_local = leg_frame_to_ik_plane(leg, foot_leg)
    return raw_ik_math(*foot_local, knee=KNEE_MODE)


NEUTRAL_MATH_FULL = {
    leg: body_ik_math(leg, STANDING_FOOT_BODY[leg])
    for leg in LEGS
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
# TRAJECTORY / GAIT
# =============================================================================

def smoothstep(u: float):
    """Smooth 0->1 interpolation."""
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def foot_trajectory_body(leg: str, phase: float):
    """Generate foot-center target for one leg.

    Stable crawl version:
        phase < SWING_RATIO:
            swing phase, foot moves forward and lifts.
        else:
            stance phase, foot remains on ground and returns slowly.
    """

    x0, y0, z0 = STANDING_FOOT_BODY[leg]

    if phase < SWING_RATIO:
        # Swing phase, normalized 0 -> 1.
        u = phase / SWING_RATIO
        us = smoothstep(u)

        # Move from back to front.
        x_back = x0 - FORWARD_SIGN_X * STRIDE_LENGTH / 2.0
        x_front = x0 + FORWARD_SIGN_X * STRIDE_LENGTH / 2.0
        x = x_back + (x_front - x_back) * us

        # Smooth lift.
        z = z0 + STEP_HEIGHT * math.sin(math.pi * u)

    else:
        # Stance phase, normalized 0 -> 1.
        u = (phase - SWING_RATIO) / (1.0 - SWING_RATIO)
        us = smoothstep(u)

        # Move from front back to rear slowly.
        x_front = x0 + FORWARD_SIGN_X * STRIDE_LENGTH / 2.0
        x_back = x0 - FORWARD_SIGN_X * STRIDE_LENGTH / 2.0
        x = x_front + (x_back - x_front) * us

        z = z0

    y = y0
    return (x, y, z)


def get_leg_phase(t: float, leg: str):
    return (t * STEP_FREQUENCY + PHASE_OFFSET[leg]) % 1.0


def current_swing_leg(t: float):
    """Return the leg currently in swing, or None."""
    for leg in LEGS:
        phase = get_leg_phase(t, leg)
        if phase < SWING_RATIO:
            return leg
    return None


def body_shift_for_swing_leg(swing_leg):
    """Approximate body/COM shift.

    In real robot control, we would shift body/root pose.
    In this simple test, we offset foot targets in the opposite direction.

    If a leg is swinging, body should lean toward the other/support legs.
    """

    if swing_leg is None:
        return (0.0, 0.0)

    # Body frame signs:
    # x: front is -X, rear is +X
    # y: left is -Y, right is +Y
    #
    # If FL swings, body should shift away from FL: slightly rear/right.
    desired_body_shift = {
        "FL": (+BODY_SHIFT_X, +BODY_SHIFT_Y),
        "FR": (+BODY_SHIFT_X, -BODY_SHIFT_Y),
        "RL": (-BODY_SHIFT_X, +BODY_SHIFT_Y),
        "RR": (-BODY_SHIFT_X, -BODY_SHIFT_Y),
    }[swing_leg]

    # Since foot targets are in body frame, shifting body one way is approximated
    # by shifting all feet the opposite way.
    return (-desired_body_shift[0], -desired_body_shift[1])


def apply_body_shift_to_foot_target(foot_body_mm, shift_xy):
    x, y, z = foot_body_mm
    sx, sy = shift_xy
    return (x + sx, y + sy, z)


def set_prim_world_pos_m(prim_path: str, pos_m):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

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

    # Optional visual markers for foot targets.
    colors = {
        "FL": (0.0, 1.0, 0.0),
        "FR": (0.0, 0.6, 1.0),
        "RL": (1.0, 0.5, 0.0),
        "RR": (1.0, 0.0, 1.0),
    }

    root_z_guess_m = 0.120

    for leg in LEGS:
        cfg = sim_utils.MeshCuboidCfg(
            size=(0.010, 0.010, 0.010),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[leg]),
        )
        x, y, z = STANDING_FOOT_BODY[leg]
        cfg.func(
            f"/World/Target_{leg}",
            cfg,
            translation=(x / 1000.0, y / 1000.0, root_z_guess_m + z / 1000.0),
        )

    return spider


# =============================================================================
# MAIN
# =============================================================================

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[0.8, 0.8, 0.5],
        target=[0.0, 0.0, 0.08],
    )

    spider = design_scene()
    sim.reset()

    print("[INFO]: Setup complete.")
    print(f"[INFO]: Joint names: {spider.joint_names}")
    print(f"[INFO]: Body names: {spider.body_names}")
    print("[INFO]: Gait Test 2: stable crawl gait.")

    default_root_state = spider.data.default_root_state.clone()
    default_joint_pos = spider.data.default_joint_pos.clone()
    default_joint_vel = spider.data.default_joint_vel.clone()

    joint_ids = {
        leg: {
            "coxa": spider.find_joints(f"coxa_{leg}_joint")[0][0],
            "femur": spider.find_joints(f"femur_{leg}_joint")[0][0],
            "tibia": spider.find_joints(f"tibia_{leg}_joint")[0][0],
        }
        for leg in LEGS
    }

    COXA_MIN, COXA_MAX = -0.70, 0.70
    FEMUR_MIN, FEMUR_MAX = -1.10, 1.10
    TIBIA_MIN, TIBIA_MAX = -1.20, 1.20

    q_smooth = default_joint_pos.clone()

    shift_smooth = np.array([0.0, 0.0], dtype=np.float64)
    SHIFT_ALPHA = 0.015

    count = 0

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        t = count * dt

        if count % 4000 == 0:
            spider.write_root_pose_to_sim(default_root_state[:, :7])
            spider.write_root_velocity_to_sim(default_root_state[:, 7:])
            spider.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            spider.reset()
            q_smooth = default_joint_pos.clone()
            shift_smooth = np.array([0.0, 0.0], dtype=np.float64)
            print("\n[INFO]: Robot reset to cfg init_state.")

        joint_target = default_joint_pos.clone()
        root_world_m = spider.data.root_pos_w[0].detach().cpu().numpy()

        swing_leg = current_swing_leg(t)

        target_shift = np.array(
            body_shift_for_swing_leg(swing_leg),
            dtype=np.float64,
        )

        shift_smooth = (1.0 - SHIFT_ALPHA) * shift_smooth + SHIFT_ALPHA * target_shift

        shift_xy = (
            float(shift_smooth[0]),
            float(shift_smooth[1]),
        )

        debug_lines = []

        for leg in LEGS:
            phase = get_leg_phase(t, leg)
            foot_body_mm = foot_trajectory_body(leg, phase)

            # Apply approximate body-shift compensation to all feet.
            foot_body_mm = apply_body_shift_to_foot_target(foot_body_mm, shift_xy)

            marker_world_m = root_world_m + np.array(foot_body_mm, dtype=np.float64) / 1000.0
            set_prim_world_pos_m(f"/World/Target_{leg}", marker_world_m)

            try:
                q, clamped = spider_ik_body(leg, *foot_body_mm)

                q[0] = clamp_joint(q[0], COXA_MIN, COXA_MAX)
                q[1] = clamp_joint(q[1], FEMUR_MIN, FEMUR_MAX)
                q[2] = clamp_joint(q[2], TIBIA_MIN, TIBIA_MAX)

                ids = joint_ids[leg]

                joint_target[:, ids["coxa"]] = (1.0 - ALPHA) * q_smooth[:, ids["coxa"]] + ALPHA * q[0]
                joint_target[:, ids["femur"]] = (1.0 - ALPHA) * q_smooth[:, ids["femur"]] + ALPHA * q[1]
                joint_target[:, ids["tibia"]] = (1.0 - ALPHA) * q_smooth[:, ids["tibia"]] + ALPHA * q[2]

                if count % 200 == 0:
                    phase_name = "SWING" if phase < SWING_RATIO else "STANCE"
                    clamp_text = " CLAMP" if clamped else ""
                    debug_lines.append(
                        f"{leg}:{phase:.2f}-{phase_name}{clamp_text} "
                        f"q=({math.degrees(q[0]):.1f},{math.degrees(q[1]):.1f},{math.degrees(q[2]):.1f})"
                    )

            except Exception as e:
                if count % 200 == 0:
                    debug_lines.append(f"{leg}: IK_ERROR {e}")

        q_smooth = joint_target.clone()

        if count % 200 == 0:
            print(
                f"[GAIT2] swing={swing_leg} shift=({shift_xy[0]:.1f},{shift_xy[1]:.1f}) | "
                + " | ".join(debug_lines)
            )

        spider.set_joint_position_target(joint_target)

        spider.write_data_to_sim()
        sim.step()
        spider.update(dt)

        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
