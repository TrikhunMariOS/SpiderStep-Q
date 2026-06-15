# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Gait-only validation in the real simulator (residual = 0).

Drives N real robots with SpiderGaitEngine to confirm the batched gait actually walks
in PhysX, the joint name->index mapping is correct, and which world direction a forward
command produces. The reference baseline that the RL residual sits on top of.

Usage:
    isaaclab.bat -p scripts/MY_Final/rl/play_gait_only.py --num_envs 16
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gait-only batched player (Phase 0-B).")
parser.add_argument("--num_envs", type=int, default=16, help="Number of robots to spawn.")
# velocity command in test5 ENGINE units: mm/s, mm/s, deg/s  (front = -X in test5 frame)
parser.add_argument("--vx", type=float, default=80.0, help="cmd vx (mm/s, test5 frame).")
parser.add_argument("--vy", type=float, default=0.0,  help="cmd vy (mm/s, test5 frame).")
parser.add_argument("--wz", type=float, default=0.0,  help="cmd wz (deg/s).")
parser.add_argument("--max_steps", type=int, default=0,
                    help="stop after N physics steps (0 = run forever). Use for a quick headless check.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# imports that need the app running
# ---------------------------------------------------------------------------
import os
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# make `gait_torch` (this folder) and `robot` (parent MY_Final folder) importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from gait_torch import SpiderGaitEngine, JOINT_NAMES_FLAT
from robot.spider_robot_cfg import SPIDER_CFG


# ---------------------------------------------------------------------------
# match test5's startup behaviour
# ---------------------------------------------------------------------------
SETTLE_STEPS   = 100      # hold default pose first (let the robot settle on the ground)
RESET_INTERVAL = 2000     # periodic reset, like test5
ALPHA          = 0.12     # joint-target EMA (test5's smoothing; keeps startup gentle)


@configclass
class SpiderSceneCfg(InteractiveSceneCfg):
    """N spider robots on a flat plane (one robot per env)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    spider = SPIDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _rpy_from_quat(quat_wxyz: torch.Tensor):
    """Batched roll/pitch/yaw (rad) from quaternion [N,4] in (w,x,y,z)."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    roll  = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw   = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def run(sim: SimulationContext, scene: InteractiveScene):
    robot = scene["spider"]
    device = robot.data.root_pos_w.device
    dt = sim.get_physics_dt()
    N = scene.num_envs

    # --- map engine joint order -> articulation joint indices (DO NOT assume) ---
    joint_ids, joint_names = robot.find_joints(JOINT_NAMES_FLAT, preserve_order=True)
    print(f"[INFO] resolved {len(joint_ids)} joints in engine order:")
    print("       ", list(zip(JOINT_NAMES_FLAT, joint_ids)))

    # --- build the batched gait engine on the sim device ---
    eng = SpiderGaitEngine(num_envs=N, device=device, dtype=torch.float32)

    # --- per-env command (engine units: mm/s, mm/s, deg/s) ---
    cmd = torch.zeros(N, 3, device=device)
    cmd[:, 0] = args_cli.vx
    cmd[:, 1] = args_cli.vy
    cmd[:, 2] = args_cli.wz
    print(f"[INFO] command (test5 frame): vx={args_cli.vx} vy={args_cli.vy} mm/s, wz={args_cli.wz} deg/s")

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()

    # EMA buffer for the 12 leg joints (in engine order), like test5's q_smooth
    q_smooth = default_joint_pos[:, joint_ids].clone()

    count = 0
    spawn_xy = scene.env_origins[:, :2].clone()   # to measure displacement per env
    last = {"dx": 0.0, "dy": 0.0, "h": 0.0, "roll": 0.0, "pitch": 0.0}

    while simulation_app.is_running():
        if args_cli.max_steps and count >= args_cli.max_steps:
            print("\n[SUMMARY] after {} steps:  meanH={:.1f}mm  maxRoll=±{:.1f}°  maxPitch=±{:.1f}°"
                  .format(count, last["h"] * 1000, last["roll"], last["pitch"]))
            print("[SUMMARY] mean world displacement: dx={:+.1f}mm dy={:+.1f}mm".format(
                last["dx"] * 1000, last["dy"] * 1000))
            print("[SUMMARY] -> if it stayed upright (roll/pitch small) and moved, the engine drives PhysX correctly.")
            break
        steps_in_cycle = count % RESET_INTERVAL

        # ---------------- periodic reset (like test5) ----------------
        if steps_in_cycle == 0:
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            scene.reset()
            q_smooth = default_joint_pos[:, joint_ids].clone()
            print(f"\n[RESET] count={count}")

        # ---------------- settle: hold default pose ----------------
        if steps_in_cycle < SETTLE_STEPS:
            robot.set_joint_position_target(default_joint_pos)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            count += 1
            continue

        # ---------------- gait clock (shared across envs here) ----------------
        t_gait = (steps_in_cycle - SETTLE_STEPS) * dt
        t = torch.full((N,), t_gait, device=device)

        # ---------------- engine: procedural gait joint targets (residual = 0) ----------------
        q_eng = eng.joint_targets(t, cmd, flat=True)            # [N,12] engine order

        # EMA smoothing (test5's ALPHA) -> gentle, matches the proven gait
        q_smooth = (1 - ALPHA) * q_smooth + ALPHA * q_eng
        robot.set_joint_position_target(q_smooth, joint_ids=joint_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        count += 1

        # ---------------- diagnostics ----------------
        if count % 200 == 0:
            pos = robot.data.root_pos_w                          # [N,3]
            roll, pitch, yaw = _rpy_from_quat(robot.data.root_quat_w)
            disp = pos[:, :2] - spawn_xy                         # [N,2] displacement
            dx = disp[:, 0].mean().item()
            dy = disp[:, 1].mean().item()
            h  = pos[:, 2].mean().item()
            last.update(dx=dx, dy=dy, h=h,
                        roll=roll.abs().max().item() * 57.3,
                        pitch=pitch.abs().max().item() * 57.3)
            print(
                f"[T {t_gait:5.1f}s] meanH={h*1000:5.1f}mm "
                f"roll=±{roll.abs().max().item()*57.3:4.1f}° "
                f"pitch=±{pitch.abs().max().item()*57.3:4.1f}° | "
                f"disp(world): dx={dx*1000:+6.1f}mm dy={dy*1000:+6.1f}mm "
                f"(>0 = +X / +Y world)"
            )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.0025, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.2, 1.2, 0.6], target=[0.0, 0.0, 0.08])

    scene = InteractiveScene(SpiderSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0))
    sim.reset()

    print("[INFO] play_gait_only — batched test5 gait via gait_torch (residual=0).")
    print(f"[INFO] num_envs={args_cli.num_envs}  device={sim.device}  dt={sim.get_physics_dt()}")
    run(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
