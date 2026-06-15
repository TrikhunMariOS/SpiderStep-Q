# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-numpy reference of the test5 gait + IK (no isaac/omni imports).

The math of OrientationCompensation_test5.py extracted so it can run in a lightweight
unit test. This is the ground truth that gait_torch.py must reproduce bit-for-bit
(~1e-6 in float64). The velocity command is passed in as cmd=(vx,vy,wz) (per-env in RL).
Units: mm, rad out of IK, deg/s for yaw.
"""

import math
import numpy as np


# =============================================================================
# ROBOT / IK PARAMETERS   (verbatim from test5)
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
# GAIT PARAMETERS   (verbatim from test5)
# =============================================================================

STEP_FREQUENCY = 1.2      # Hz
# control-point z, not the actual apex (the cubic Bezier doesn't pass through it):
# 95 gives a real swing apex ~52mm so the gait clears 50mm obstacles on its own
STEP_HEIGHT    = 95.0     # mm

MAX_STRIDE_LENGTH = 100.0
TURN_GAIN         = 0.55

SWING_RATIO = 0.25

BEZIER_LIFT_FRONT_RATIO = 0.15
BEZIER_LIFT_BACK_RATIO  = 0.65
# late-swing control-point z (fraction of STEP_HEIGHT): 0.40 keeps the foot up later in
# swing so it stays clear while crossing a box edge
BEZIER_LAND_RATIO       = 0.40

PHASE_OFFSET = {
    "FL": 0.000,
    "RR": 0.250,
    "FR": 0.500,
    "RL": 0.750,
}

# Per-leg asymmetric joint limits (deg -> rad)
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
# FEEDFORWARD CoG PARAMETERS  (verbatim from test5)
# =============================================================================

ENABLE_COG_SHIFT  = True
COG_SHIFT_GAIN    = 0.30
COG_LEAD          = 0.06
COG_RAMP_TIME     = 0.6
COG_MAX_OFFSET_MM = 30.0
STANCE_RAMP       = 0.16

# Tilt comp is DISABLED in test5 (ENABLE_TILT_COMP = False); kept here only as a
# documented constant.  The active control path uses CoG shift only.
ENABLE_TILT_COMP = False


# =============================================================================
# IK FUNCTIONS   (verbatim from test5)
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
# GAIT TRAJECTORY   (verbatim from test5, with cmd passed in)
# =============================================================================

def command_to_stride_vector_mm(cmd):
    """cmd = (vx_mm_s, vy_mm_s, wz_deg_s).  Returns (sx, sy) stride in mm."""
    cmd_vx, cmd_vy, _ = cmd
    gait_period = 1.0 / max(STEP_FREQUENCY, 1e-6)
    sx = -cmd_vx * gait_period
    sy = -cmd_vy * gait_period
    mag = math.sqrt(sx**2 + sy**2)
    if mag > MAX_STRIDE_LENGTH:
        s = MAX_STRIDE_LENGTH / max(mag, 1e-6)
        sx *= s; sy *= s
    return sx, sy


def turning_offset_for_leg(leg, cmd):
    cmd_wz = cmd[2]
    if abs(cmd_wz) < 1e-6:
        return 0.0, 0.0
    x0, y0, _  = GAIT_FOOT_BODY[leg]
    wz          = math.radians(cmd_wz)
    gait_period = 1.0 / max(STEP_FREQUENCY, 1e-6)
    return (-wz * y0 * gait_period * TURN_GAIN,
             wz * x0 * gait_period * TURN_GAIN)


def smoothstep(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def bezier3(p0, p1, p2, p3, u):
    u = max(0.0, min(1.0, u)); v = 1.0 - u
    return v**3*p0 + 3*v**2*u*p1 + 3*v*u**2*p2 + u**3*p3


def foot_trajectory_body(leg, phase, cmd):
    x0, y0, z0       = GAIT_FOOT_BODY[leg]
    cmd_dx, cmd_dy   = command_to_stride_vector_mm(cmd)
    turn_dx, turn_dy = turning_offset_for_leg(leg, cmd)
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
                       z0+STEP_HEIGHT*BEZIER_LAND_RATIO])
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
# STANCE WEIGHT + FEEDFORWARD CoG  (verbatim from test5)
# =============================================================================

def stance_weight(leg_phase):
    p = leg_phase % 1.0
    if p < SWING_RATIO:
        return 0.0
    up   = smoothstep((p - SWING_RATIO) / STANCE_RAMP)
    down = smoothstep((1.0 - p)         / STANCE_RAMP)
    return min(up, down)


def compute_cog_offset(t):
    """Feedforward body lean (mm).  Pure function of gait time t (does not
    depend on the velocity command)."""
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


def apply_cog_shift(foot_body_mm, cog_x, cog_y):
    """Shifting every foot by -cog == translating the body by +cog.
    (Tilt comp is disabled in test5, so this is the full correction.)"""
    x, y, z = foot_body_mm
    if ENABLE_COG_SHIFT:
        x -= cog_x
        y -= cog_y
    return (x, y, z)


# =============================================================================
# FULL PIPELINE HELPERS  (for the validation script)
# =============================================================================

def numpy_ik_body_array(foot_mm_4x3):
    """foot_mm_4x3 : ndarray [4,3] in LEGS order -> joint targets [4,3] (rad),
    sign + neutral applied + joint-limit clamped (exactly test5's output)."""
    out = np.zeros((4, 3), dtype=np.float64)
    for i, leg in enumerate(LEGS):
        x, y, z = foot_mm_4x3[i]
        q, _ = spider_ik_body(leg, float(x), float(y), float(z))
        q = clamp_joint_leg(leg, q[0], q[1], q[2])
        out[i] = q
    return out


def numpy_foot_targets(t, cmd):
    """Return foot targets [4,3] in body frame (mm) AFTER CoG shift, exactly as
    test5 commands them (with ENABLE_TILT_COMP = False)."""
    cog_x, cog_y = compute_cog_offset(t) if ENABLE_COG_SHIFT else (0.0, 0.0)
    out = np.zeros((4, 3), dtype=np.float64)
    for i, leg in enumerate(LEGS):
        phase    = get_leg_phase(t, leg)
        foot_nom = foot_trajectory_body(leg, phase, cmd)
        out[i]   = apply_cog_shift(foot_nom, cog_x, cog_y)
    return out


def numpy_joint_targets(t, cmd):
    """Full pipeline: gait time t + command -> joint targets [4,3] (rad)."""
    return numpy_ik_body_array(numpy_foot_targets(t, cmd))
