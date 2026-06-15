# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phoenix-style balance gait — the proven procedural gait (ground truth).

Fixes the ±13° body roll ("wobble") of earlier versions with three coupled ideas
from the Lynxmotion Phoenix gait engine:

  A) Feedforward, phase-locked CoG shift: body lean is a pure function of gait phase
     (no laggy EMA), evaluated slightly ahead (COG_LEAD) so the body is already over
     the next support triangle before a leg lifts. A slow walker stays up only while
     the CoM projects inside the stance-feet polygon, so this pre-shift is essential.
  B) Perimeter lift order (FL->RL->RR->FR): walks the support centroid smoothly around
     the rectangle without crossing center, so the body sways cleanly instead of wobbling.
  C) Tilt compensation on stance legs only: applying it to swing legs created a roll
     feedback loop; stance-only scaling breaks it.

Plus duty factor 0.80 (brief all-4-down phase between swings).

Usage:
    isaaclab.bat -p OrientationCompensation_test5.py
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OrientationCompensation_test5 for Spider Robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, Gf
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from robot.spider_robot_cfg import SPIDER_CFG


# =============================================================================
# ROBOT / IK PARAMETERS   (unchanged from test3)
# =============================================================================

L1 = 35.0
L2 = 65.0
L3 = 100.0

FOOT_RADIUS = 10.0
KNEE_MODE   = "down"

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

ZERO_FOOT_CONTACT_BODY = {
    "FL": (-161.0, -113.0, -76.3),
    "FR": (-161.0,  113.0, -76.3),
    "RL": ( 161.0, -113.0, -76.3),
    "RR": ( 161.0,  113.0, -76.3),
}

ZERO_FOOT_BODY = {
    leg: (x, y, z + FOOT_RADIUS)
    for leg, (x, y, z) in ZERO_FOOT_CONTACT_BODY.items()
}

GAIT_FOOT_CONTACT_BODY = {
    "FL": (-160.0, -110.0, -66.34),
    "FR": (-160.0,  110.0, -66.34),
    "RL": ( 160.0, -110.0, -66.34),
    "RR": ( 160.0,  110.0, -66.34),
}

GAIT_FOOT_BODY = {
    leg: (x, y, z + FOOT_RADIUS)
    for leg, (x, y, z) in GAIT_FOOT_CONTACT_BODY.items()
}


# =============================================================================
# GAIT PARAMETERS
# =============================================================================

STEP_FREQUENCY = 1.2      # Hz  (test3 used 1.5; slower = better swing tracking)
STEP_HEIGHT    = 70.0     # mm

# Velocity command.  Convention: front = -X, left = -Y.
# CMD_VX/VY in mm/s, CMD_WZ in deg/s (positive = CCW from top)
CMD_VX = 70.0
CMD_VY = 0.0
CMD_WZ = 0.0

MAX_STRIDE_LENGTH = 100.0
TURN_GAIN         = 0.55

# ── Duty factor ────────────────────────────────────────────────────────────
# SWING_RATIO = fraction of the cycle a leg is in the air.
# 0.25 -> duty 0.75 -> exactly ONE leg swings at a time and exactly THREE are
# always down (no all-4-down beat).  Crucial here: it means the support-centroid
# never collapses to body-centre, so the feedforward CoG has no discontinuity.
SWING_RATIO = 0.25

BEZIER_LIFT_FRONT_RATIO = 0.15
BEZIER_LIFT_BACK_RATIO  = 0.65
# Soft-landing: height (fraction of STEP_HEIGHT) of the 2nd swing control point.
# 1.0 = old flat-top arch -> foot slams down at touchdown (velocity ~ -3*H).
# 0.25 = foot peaks early then DESCENDS GENTLY -> touchdown velocity ~ -0.75*H
# (≈4x softer) so the underdamped joints aren't shock-excited at foot strike.
BEZIER_LAND_RATIO = 0.25

# ── Lift order: DIAGONAL (test3's proven-stable order) ──────────────────────
# FL -> RR -> FR -> RL alternates BOTH front/rear AND left/right every step, so
# the body is balanced fore-aft and laterally at all times -> statically stable
# even with a weak CoG shift.
# (A "perimeter" order FL->RL->RR->FR gives a prettier CoG circle BUT lifts two
#  rear legs (and two left legs) back-to-back -> pitches/rolls over -> it blew up.
#  Lesson: balance of the lift order beats prettiness of the CoG path.)
PHASE_OFFSET = {
    "FL": 0.000,
    "RR": 0.250,
    "FR": 0.500,
    "RL": 0.750,
}

ALPHA = 0.12              # joint-target EMA smoothing (on the IK output, not CoG)

# ── Per-leg asymmetric joint limits (deg -> rad) ── (unchanged from test3) ───
JOINT_LIMITS = {
    "FL": {"coxa": (-math.radians(75),  math.radians(50)),
           "femur": (-math.radians(110), math.radians(80)),
           "tibia": (-math.radians(75),  math.radians(110))},
    "FR": {"coxa": (-math.radians(50),  math.radians(75)),
           "femur": (-math.radians(80),  math.radians(110)),
           "tibia": (-math.radians(110), math.radians(75))},
    "RL": {"coxa": (-math.radians(50),  math.radians(75)),
           "femur": (-math.radians(80),  math.radians(110)),
           "tibia": (-math.radians(110), math.radians(75))},
    "RR": {"coxa": (-math.radians(75),  math.radians(50)),
           "femur": (-math.radians(110), math.radians(80)),
           "tibia": (-math.radians(75),  math.radians(110))},
}


# =============================================================================
# STABILITY PARAMETERS
# =============================================================================

SETTLE_STEPS   = 100
RESET_INTERVAL = 2000

# ── Feedforward phase-locked CoG — support-centroid (idea A) ────────────────
ENABLE_COG_SHIFT = True
# The body leans toward the centroid of the SUPPORTING feet, computed straight
# from gait PHASE (feedforward, no EMA -> no lag) and evaluated COG_LEAD ahead so
# the lean precedes the lift.  The centroid automatically points away from the
# swinging leg for ANY lift order (no hand-tuned angle).  With duty 0.75 it never
# collapses to centre.  A startup ramp avoids the impulsive kick.
COG_SHIFT_GAIN    = 0.30   # fraction of centroid offset (~65mm) -> ~20mm lean
COG_LEAD          = 0.06   # phase lead (fraction of cycle)
COG_RAMP_TIME     = 0.6    # s — ramp the shift in after settle
COG_MAX_OFFSET_MM = 30.0   # safety clamp
# Width (in phase) of the stance-weight ramp (CoG centroid + TILT fade).
# Wider = the support-centroid (hence the body CoG shift) transitions more
# GRADUALLY at each leg hand-off -> smoother body sway, with ZERO added lag
# (still feedforward).  0.06 was abrupt; 0.12 rounds the hand-offs.
STANCE_RAMP = 0.16

# ── Tilt compensation, stance-only (idea C) ─────────────────────────────────
ENABLE_TILT_COMP = False
TILT_XY_GAIN = 0.50     # kinematic frame correction
TILT_Z_GAIN  = 0.60     # active leveling (gentler than test3's 0.9 — feedforward
                        # CoG now does most of the work, tilt just trims)
TILT_MAX_DEG = 20.0


# =============================================================================
# IK FUNCTIONS   (verbatim from test3 — proven correct)
# =============================================================================

def body_to_leg_frame(leg, foot_body_mm):
    fx, fy, fz = foot_body_mm
    hx, hy, hz = HIP_POS_BODY[leg]
    return fx - hx, fy - hy, fz - hz


def leg_frame_to_ik_plane(leg, foot_leg_mm):
    x, y, z   = foot_leg_mm
    sx, sy, _ = body_to_leg_frame(leg, ZERO_FOOT_BODY[leg])
    base_angle = math.atan2(sy, sx)
    c = math.cos(-base_angle)
    s = math.sin(-base_angle)
    return c * x - s * y, s * x + c * y, z


def raw_ik_math(x_mm, y_mm, z_mm, knee="down"):
    theta1     = math.atan2(y_mm, x_mm)
    horizontal = math.sqrt(x_mm**2 + y_mm**2)
    r          = horizontal - L1
    D          = math.sqrt(r**2 + z_mm**2)

    safe_min = abs(L2 - L3) + 5.0
    safe_max = L2 + L3      - 5.0
    clamped  = False
    if D > safe_max:
        scale = safe_max / max(D, 1e-6)
        r *= scale; z_mm *= scale; D = safe_max; clamped = True
    elif D < safe_min:
        scale = safe_min / max(D, 1e-6)
        r *= scale; z_mm *= scale; D = safe_min; clamped = True

    cos_alpha = (L2**2 + D**2  - L3**2) / (2 * L2 * D)
    cos_beta  = (L2**2 + L3**2 - D**2)  / (2 * L2 * L3)
    alpha = math.acos(np.clip(cos_alpha, -1.0, 1.0))
    beta  = math.acos(np.clip(cos_beta,  -1.0, 1.0))
    base  = math.atan2(z_mm, r)

    if knee == "down":
        theta2 = base + alpha
        theta3 = math.pi - beta
    else:
        theta2 = base - alpha
        theta3 = -(math.pi - beta)
    return theta1, theta2, theta3, clamped


def body_ik_math(leg, foot_body_mm):
    return raw_ik_math(*leg_frame_to_ik_plane(leg, body_to_leg_frame(leg, foot_body_mm)),
                       knee=KNEE_MODE)


NEUTRAL_MATH_FULL = {leg: body_ik_math(leg, ZERO_FOOT_BODY[leg]) for leg in LEGS}
NEUTRAL_MATH      = {leg: v[:3] for leg, v in NEUTRAL_MATH_FULL.items()}


def spider_ik_body(leg, x_mm, y_mm, z_mm):
    q0, q1, q2, clamped = body_ik_math(leg, (x_mm, y_mm, z_mm))
    qn   = NEUTRAL_MATH[leg]
    sign = LEG_SIGN[leg]
    return [sign["coxa"]  * (q0 - qn[0]),
            sign["femur"] * (q1 - qn[1]),
            sign["tibia"] * (q2 - qn[2])], clamped


def clamp_joint_leg(leg, q_coxa, q_femur, q_tibia):
    lims = JOINT_LIMITS[leg]
    return [max(min(q_coxa,  lims["coxa"][1]),  lims["coxa"][0]),
            max(min(q_femur, lims["femur"][1]), lims["femur"][0]),
            max(min(q_tibia, lims["tibia"][1]), lims["tibia"][0])]


# =============================================================================
# GAIT TRAJECTORY   (unchanged from test3)
# =============================================================================

def command_to_stride_vector_mm():
    gait_period = 1.0 / max(STEP_FREQUENCY, 1e-6)
    sx = -CMD_VX * gait_period
    sy = -CMD_VY * gait_period
    mag = math.sqrt(sx**2 + sy**2)
    if mag > MAX_STRIDE_LENGTH:
        s = MAX_STRIDE_LENGTH / max(mag, 1e-6)
        sx *= s; sy *= s
    return sx, sy


def turning_offset_for_leg(leg):
    if abs(CMD_WZ) < 1e-6:
        return 0.0, 0.0
    x0, y0, _  = GAIT_FOOT_BODY[leg]
    wz          = math.radians(CMD_WZ)
    gait_period = 1.0 / max(STEP_FREQUENCY, 1e-6)
    return (-wz * y0 * gait_period * TURN_GAIN,
             wz * x0 * gait_period * TURN_GAIN)


def smoothstep(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def bezier3(p0, p1, p2, p3, u):
    u = max(0.0, min(1.0, u)); v = 1.0 - u
    return v**3*p0 + 3*v**2*u*p1 + 3*v*u**2*p2 + u**3*p3


def foot_trajectory_body(leg, phase):
    x0, y0, z0       = GAIT_FOOT_BODY[leg]
    cmd_dx, cmd_dy   = command_to_stride_vector_mm()
    turn_dx, turn_dy = turning_offset_for_leg(leg)
    dx = cmd_dx + turn_dx
    dy = cmd_dy + turn_dy

    if math.sqrt(dx**2 + dy**2) < 1e-6:
        return (x0, y0, z0)

    xb, yb = x0 - dx/2, y0 - dy/2
    xf, yf = x0 + dx/2, y0 + dy/2

    if phase < SWING_RATIO:
        u  = phase / SWING_RATIO
        p0 = np.array([xb, yb, z0])
        p1 = np.array([xb+(xf-xb)*BEZIER_LIFT_FRONT_RATIO,
                       yb+(yf-yb)*BEZIER_LIFT_FRONT_RATIO, z0+STEP_HEIGHT])
        p2 = np.array([xb+(xf-xb)*BEZIER_LIFT_BACK_RATIO,
                       yb+(yf-yb)*BEZIER_LIFT_BACK_RATIO,
                       z0+STEP_HEIGHT*BEZIER_LAND_RATIO])   # soft landing
        p3 = np.array([xf, yf, z0])
        p  = bezier3(p0, p1, p2, p3, u)
        return (float(p[0]), float(p[1]), float(p[2]))
    else:
        u  = (phase - SWING_RATIO) / (1.0 - SWING_RATIO)
        us = smoothstep(u)
        return (xf + (xb-xf)*us, yf + (yb-yf)*us, z0)


def get_leg_phase(t, leg):
    return (t * STEP_FREQUENCY + PHASE_OFFSET[leg]) % 1.0


# =============================================================================
# STANCE WEIGHT  —  shared by the CoG feedforward AND the stance-only tilt.
# =============================================================================

def stance_weight(leg_phase):
    """Smooth 'how much is this leg supporting' in [0,1].

    0 during swing (phase < SWING_RATIO), ramps to 1 just after touchdown,
    holds 1, ramps back to 0 just before the next lift-off.  The smoothness is
    SPATIAL (a function of phase), so the centroid built from these weights is
    continuous WITHOUT any laggy temporal filter — that is the key difference
    from test3's EMA.
    """
    p = leg_phase % 1.0
    if p < SWING_RATIO:
        return 0.0
    up   = smoothstep((p - SWING_RATIO) / STANCE_RAMP)   # fade in after landing
    down = smoothstep((1.0 - p)         / STANCE_RAMP)   # fade out before lift
    return min(up, down)


# =============================================================================
# FEEDFORWARD PHASE-LOCKED CoG SHIFT  (idea A)
# =============================================================================

def compute_cog_offset(t):
    """Feedforward body lean = COG_SHIFT_GAIN * (centroid of supporting feet),
    evaluated COG_LEAD ahead of the gait phase, ramped in after settle.

    Pure function of phase -> phase-locked, zero lag, leads the lift.  The
    centroid of the 3 stance feet sits to the side AWAY from the swinging leg, so
    leaning toward it gives the correct compensation for any lift order with no
    hand-tuned angle.  With duty 0.75 there are always ~3 stance feet, so the
    centroid never collapses to body-centre.
    """
    cx_num = cy_num = wsum = 0.0
    for leg in LEGS:
        p = (t * STEP_FREQUENCY + PHASE_OFFSET[leg] + COG_LEAD) % 1.0
        w = stance_weight(p)
        x, y, _ = GAIT_FOOT_BODY[leg]
        cx_num += w * x; cy_num += w * y; wsum += w

    if wsum < 1e-6:
        return 0.0, 0.0

    ramp = min(1.0, max(0.0, t / max(COG_RAMP_TIME, 1e-6)))
    tx = ramp * COG_SHIFT_GAIN * (cx_num / wsum)
    ty = ramp * COG_SHIFT_GAIN * (cy_num / wsum)

    mag = math.hypot(tx, ty)
    if mag > COG_MAX_OFFSET_MM:
        f = COG_MAX_OFFSET_MM / mag
        tx *= f; ty *= f
    return tx, ty


#   =============================================================================
#    TILT COMPENSATION  (idea C: scaled by stance weight, so swing legs get none)
#   =============================================================================

def apply_corrections(foot_body_mm, cog_x, cog_y, roll_rad, pitch_rad, w_stance):
    """1) body translation (CoG shift) — applied to ALL legs (it is body motion);
       2) tilt leveling — applied only in proportion to w_stance (0 for swing).
    """
    x, y, z = foot_body_mm

    # --- CoG shift: shifting every foot by -cog == translating the body by +cog
    if ENABLE_COG_SHIFT:
        x -= cog_x
        y -= cog_y

    if not ENABLE_TILT_COMP or w_stance <= 1e-3:
        return (x, y, z)

    # --- Tilt: rotate the (already CoG-shifted) target by R_tilt^T to cancel
    #     body roll/pitch, then blend in by the stance weight so swing legs are
    #     untouched (breaks the roll feedback loop).  Yaw intentionally ignored.
    p = np.array([x, y, z], dtype=np.float64)
    max_rad = math.radians(TILT_MAX_DEG)
    roll_c  = float(np.clip(roll_rad,  -max_rad, max_rad))
    pitch_c = float(np.clip(pitch_rad, -max_rad, max_rad))
    R_tilt  = Rotation.from_euler('xyz', [roll_c, pitch_c, 0.0]).as_matrix()
    delta   = (R_tilt.T @ p) - p

    return (float(p[0] + w_stance * TILT_XY_GAIN * delta[0]),
            float(p[1] + w_stance * TILT_XY_GAIN * delta[1]),
            float(p[2] + w_stance * TILT_Z_GAIN  * delta[2]))


# =============================================================================
# BODY ROTATION READER + USD MARKER HELPER   (unchanged from test3)
# =============================================================================

def get_body_rotation(spider):
    q = spider.data.root_quat_w[0].detach().cpu().numpy()   # [w,x,y,z]
    w, x, y, z = q
    R = Rotation.from_quat([x, y, z, w])
    roll, pitch, yaw = R.as_euler('xyz')
    return R.as_matrix(), roll, pitch, yaw


def set_prim_world_pos_m(prim_path, pos_m):
    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    ops   = [op for op in xform.GetOrderedXformOps()
             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    vec   = Gf.Vec3d(float(pos_m[0]), float(pos_m[1]), float(pos_m[2]))
    if ops:
        ops[0].Set(vec)
    else:
        xform.AddTranslateOp().Set(vec)


# =============================================================================
# SCENE
# =============================================================================

def design_scene():
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/lightDistant", light_cfg, translation=(1.0, 0.0, 10.0))

    spider_cfg = SPIDER_CFG.replace(prim_path="/World/Spider")
    spider     = Articulation(spider_cfg)

    colors = {"FL": (0.0, 1.0, 0.0), "FR": (0.0, 0.6, 1.0),
              "RL": (1.0, 0.5, 0.0), "RR": (1.0, 0.0, 1.0)}
    for leg in LEGS:
        cfg = sim_utils.MeshCuboidCfg(
            size=(0.010, 0.010, 0.010),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[leg]))
        x, y, z = GAIT_FOOT_BODY[leg]
        cfg.func(f"/World/Target_{leg}", cfg,
                 translation=(x/1000.0, y/1000.0, 0.120 + z/1000.0))
    return spider


# =============================================================================
# MAIN
# =============================================================================

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.0025, device=args_cli.device)
    sim     = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.8, 0.8, 0.5], target=[0.0, 0.0, 0.08])

    spider = design_scene()
    sim.reset()

    print("[INFO] OrientationCompensation_test5 — Phoenix-style feedforward CoG.")
    print(f"[INFO] COG: gain={COG_SHIFT_GAIN} lead={COG_LEAD} ramp={COG_RAMP_TIME}s "
          f"max={COG_MAX_OFFSET_MM}mm (feedforward centroid, no EMA)")
    print(f"[INFO] TILT(stance-only): XY={TILT_XY_GAIN} Z={TILT_Z_GAIN}")
    print(f"[INFO] gait: f={STEP_FREQUENCY}Hz swing={SWING_RATIO} order={list(PHASE_OFFSET)}")
    print(f"[INFO] CMD: vx={CMD_VX} vy={CMD_VY} wz={CMD_WZ}")

    default_root_state = spider.data.default_root_state.clone()
    default_joint_pos  = spider.data.default_joint_pos.clone()
    default_joint_vel  = spider.data.default_joint_vel.clone()

    joint_ids = {
        leg: {
            "coxa":  spider.find_joints(f"coxa_{leg}_joint")[0][0],
            "femur": spider.find_joints(f"femur_{leg}_joint")[0][0],
            "tibia": spider.find_joints(f"tibia_{leg}_joint")[0][0],
        } for leg in LEGS
    }
    foot_ids = {leg: spider.find_bodies(f"Foot_{leg}_1")[0][0] for leg in LEGS}

    q_smooth = default_joint_pos.clone()
    count    = 0

    def _fresh_amp():
        return {leg: [1e9, -1e9, 1e9, -1e9] for leg in LEGS}
    foot_amp = _fresh_amp()

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        steps_in_cycle = count % RESET_INTERVAL

        if steps_in_cycle == 0:
            spider.write_root_pose_to_sim(default_root_state[:, :7])
            spider.write_root_velocity_to_sim(default_root_state[:, 7:])
            spider.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            spider.reset()
            q_smooth = default_joint_pos.clone()
            print(f"\n[RESET] count={count}")

        if steps_in_cycle < SETTLE_STEPS:
            spider.set_joint_position_target(default_joint_pos)
            spider.write_data_to_sim()
            sim.step()
            spider.update(dt)
            count += 1
            continue

        t_gait = (steps_in_cycle - SETTLE_STEPS) * dt

        root_world_m            = spider.data.root_pos_w[0].detach().cpu().numpy()
        R_mat, roll, pitch, yaw = get_body_rotation(spider)

        leg_phases = {leg: get_leg_phase(t_gait, leg) for leg in LEGS}

        # feedforward, phase-locked CoG (no EMA)
        cog_x, cog_y = compute_cog_offset(t_gait) if ENABLE_COG_SHIFT else (0.0, 0.0)

        joint_target = default_joint_pos.clone()
        debug_lines  = []
        foot_diag    = {}

        for leg in LEGS:
            phase   = leg_phases[leg]
            w_st    = stance_weight(phase)        # 0 swing .. 1 stance

            foot_nom = foot_trajectory_body(leg, phase)
            foot_mm  = apply_corrections(foot_nom, cog_x, cog_y, roll, pitch, w_st)

            marker_m = root_world_m + R_mat @ np.array(foot_mm) / 1000.0
            set_prim_world_pos_m(f"/World/Target_{leg}", marker_m)

            # [DIAG] commanded vs actual foot Z in body frame (bob-removed)
            act_foot_w = spider.data.body_pos_w[0, foot_ids[leg]].detach().cpu().numpy()
            cmd_z_body = float(foot_mm[2])
            act_z_body = float((R_mat.T @ (act_foot_w - root_world_m))[2] * 1000.0)
            err3d      = float(np.linalg.norm(marker_m - act_foot_w) * 1000.0)
            foot_diag[leg] = (phase, err3d)
            a = foot_amp[leg]
            a[0] = min(a[0], cmd_z_body); a[1] = max(a[1], cmd_z_body)
            a[2] = min(a[2], act_z_body); a[3] = max(a[3], act_z_body)

            try:
                q, clamped = spider_ik_body(leg, *foot_mm)
                q = clamp_joint_leg(leg, q[0], q[1], q[2])
                ids = joint_ids[leg]
                for jname, qi in zip(("coxa", "femur", "tibia"), q):
                    idx = ids[jname]
                    joint_target[:, idx] = (1 - ALPHA) * q_smooth[:, idx] + ALPHA * qi
            except Exception as e:
                if count % 200 == 0:
                    debug_lines.append(f"{leg}:ERR {e}")

        q_smooth = joint_target.clone()

        if count % 200 == 0:
            print(f"[T5 {t_gait:5.1f}s] "
                  f"rpy=({math.degrees(roll):+.1f},{math.degrees(pitch):+.1f},"
                  f"{math.degrees(yaw):+.1f})° cog=({cog_x:+.1f},{cog_y:+.1f})mm "
                  + " ".join(debug_lines))
            print("   [DIAG] " + " | ".join(
                f"{leg}: cmdLift={foot_amp[leg][1]-foot_amp[leg][0]:.0f} "
                f"actLift={foot_amp[leg][3]-foot_amp[leg][2]:.0f} "
                f"err={foot_diag[leg][1]:.0f}" for leg in LEGS) + "  (mm)")
            foot_amp = _fresh_amp()

        spider.set_joint_position_target(joint_target)
        spider.write_data_to_sim()
        sim.step()
        spider.update(dt)
        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
