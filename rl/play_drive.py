# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the trained policy yourself with the keyboard — one robot, training terrain.

Drive it at the obstacle you want to test and feel how it responds.

Controls (Isaac Lab Se2Keyboard)
--------------------------------
    Arrow Up / Down      : forward / backward   (vx ±0.10 m/s)
    Arrow Right / Left   : strafe               (vy ±0.06 m/s)  *flip sign in code if inverted
    Z / X                : turn                 (wz ±0.3 rad/s)
    L                    : zero all commands (stop)
    R                    : reset the robot (respawn)
    (keys ACCUMULATE per press; L to clear)

Usage:
    isaaclab.bat -p scripts/MY_Final/rl/play_drive.py --checkpoint <path\\model_xxx.pt>
    (omit --checkpoint to auto-load the latest run/model from logs/rsl_rl/spider_gapstairs_rough)
    add --debug_rays to draw the height-scanner hit points
NOTE: needs the GUI (keyboard lives in the app window) — do NOT pass --headless.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard-drive the Spider residual policy.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Spider-Residual-Play-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model_*.pt (default: latest run)")
parser.add_argument("--debug_rays", action="store_true", default=False, help="Draw height-scanner hit points")
parser.add_argument("--follow", action="store_true", default=False, help="Camera follows the robot (default: free camera)")
parser.add_argument("--no_arrows", action="store_true", default=False, help="Hide the command/velocity arrows (green = command, blue = ACTUAL velocity — blue's direction is meaningless noise at standstill)")
parser.add_argument("--raw", action="store_true", default=False, help="RAW mode: disable the heading-hold assist -> see the policy's TRUE gait (incl. residual strafe yaw)")
parser.add_argument("--residual_scale", type=float, default=1.0, help="Scale the RL foot offset at PLAY time (NO retrain). 1.0=full RL; 0.5=half; 0.0=pure procedural gait (smoothest). Lower it for clean flat demo shots, keep 1.0 for obstacle shots.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.headless:
    raise SystemExit("[play_drive] keyboard teleop needs the GUI — remove --headless.")

# launch the app FIRST (everything below imports omni/isaaclab modules)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

# register our gym tasks + grab the cfg classes
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import rl  # noqa: F401  (gym.register)
from rl.agents.rsl_rl_ppo_cfg import (
    SpiderResidualRoughPPORunnerCfg,
    SpiderResidualFlatPPORunnerCfg,
)
from rl.rough_env_cfg import SpiderResidualRoughEnvCfg_PLAY
from rl.flat_env_cfg import SpiderResidualFlatEnvCfg_PLAY

# command envelope to clamp keyboard output to. Must match the training ranges, or
# out-of-distribution commands make the policy twitch instead of follow.
CMD_MAX = (0.14, 0.12, 0.4)   # vx m/s, vy m/s, wz rad/s

# per-axis keyboard sign (teleop only). Flip an axis here if its arrow feels reversed.
#   index 0 = forward/back | 1 = strafe | 2 = turn
KB_SIGN = (1.0, 1.0, 1.0)


def build_drive_cfg():
    """Instantiate the PLAY cfg matching --task (Flat=speed arena, else Rough) and apply
    the driving overrides.  Returns (env_cfg, agent_cfg, is_flat)."""
    is_flat = "Flat" in args_cli.task
    if is_flat:
        cfg = SpiderResidualFlatEnvCfg_PLAY()
        agent_cfg = SpiderResidualFlatPPORunnerCfg()
    else:
        cfg = SpiderResidualRoughEnvCfg_PLAY()
        agent_cfg = SpiderResidualRoughPPORunnerCfg()

    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600.0                    # no timeout while driving
    # we overwrite the command buffer every frame -> never let the term resample over us
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.debug_vis = not args_cli.no_arrows   # green=cmd, blue=actual vel

    # camera: FREE by default (easier debugging); --follow locks it to the robot
    if args_cli.follow:
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.eye = (0.9, 0.9, 0.5)
        cfg.viewer.lookat = (0.0, 0.0, 0.1)
    else:
        cfg.viewer.eye = (2.0, 2.0, 1.5)
        cfg.viewer.lookat = (0.0, 0.0, 0.1)

    # height_scanner exists ONLY on rough — guard so the flat speed arena doesn't crash
    if getattr(cfg.scene, "height_scanner", None) is not None:
        cfg.scene.height_scanner.debug_vis = args_cli.debug_rays
        # default ray marker is a 2cm sphere on a 5cm grid -> overlapping blobs.  Shrink to
        # ~6mm (point-like, ≈ foot-ball size) to read exact ray positions + see edge gaps.
        cfg.scene.height_scanner.visualizer_cfg = RAY_CASTER_MARKER_CFG.replace(
            prim_path="/Visuals/RayCaster",
            markers={"hit": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            )},
        )
    return cfg, agent_cfg, is_flat


def main():
    env_cfg, agent_cfg, is_flat = build_drive_cfg()
    print(f"[play_drive] arena: {'FLAT (speed)' if is_flat else 'ROUGH (obstacles)'}")
    # convert our (old-style) runner cfg to the installed rsl_rl version's format —
    # train.py/play.py always do this; skipping it -> KeyError: 'class_name'
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # resolve checkpoint: explicit path or latest run in this experiment's log dir
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[play_drive] loading checkpoint: {resume_path}")

    # env + rsl_rl wrapper + policy (mirrors scripts/reinforcement_learning/rsl_rl/play.py)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_has_reset = hasattr(policy, "reset")

    # keyboard: sensitivities = the trained command envelope (1 press = full speed)
    keyboard = Se2Keyboard(Se2KeyboardCfg(
        v_x_sensitivity=CMD_MAX[0],
        v_y_sensitivity=CMD_MAX[1],
        omega_z_sensitivity=CMD_MAX[2],
        sim_device=str(env.unwrapped.device),
    ))
    reset_requested = [False]
    keyboard.add_callback("R", lambda: reset_requested.__setitem__(0, True))
    print(keyboard)
    print("[play_drive] R = respawn robot, L = stop (zero commands)")

    # the command term whose buffer we overwrite each frame
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_max = torch.tensor(CMD_MAX, device=env.unwrapped.device)
    kb_sign = torch.tensor(KB_SIGN, device=env.unwrapped.device)   # per-axis direction (see KB_SIGN)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    act_term = env.unwrapped.action_manager._terms["joint_pos"]   # for gait-freq telemetry
    # play-time residual dial (no retrain): scales how far the RL deviates from the gait.
    # 1.0 = full RL (best on obstacles), 0.0 = pure gait (smoothest on flat).
    if args_cli.residual_scale != 1.0:
        _base = act_term.cfg.offset_scale_mm
        act_term.cfg.offset_scale_mm = _base * args_cli.residual_scale
        print(f"[play_drive] residual_scale={args_cli.residual_scale:.2f} -> "
              f"offset_scale_mm {_base:.1f} -> {act_term.cfg.offset_scale_mm:.1f} mm")
    step_count = 0

    # slew-rate limiter: ease the command toward the keyboard target instead of snapping,
    # so a reversal doesn't yank the gait's stride vector and rock the body
    slew_per_s = torch.tensor((1.0, 0.9, 2.7), device=env.unwrapped.device)
    cmd_now = torch.zeros(3, device=env.unwrapped.device)

    # heading-hold assist (teleop only): cancel the crawl gait's residual yaw while strafing
    # so it goes straight. If it ever spins, negate HEADING_KP.
    HEADING_KP = 4.0          # rad/s of correction per rad of heading error
    TURN_DEADZONE = 0.03      # |wz| above this = "user is turning" -> re-latch heading
    MOVE_DEADZONE = 0.02      # need real translation before holding heading (else marches in place)
    heading_hold_on = not args_cli.raw
    asset = env.unwrapped.scene["robot"]
    target_yaw = euler_xyz_from_quat(asset.data.root_quat_w)[2][0]
    print(f"[play_drive] heading-hold: {'OFF (raw gait)' if args_cli.raw else 'ON'}")

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if reset_requested[0]:
                reset_requested[0] = False
                keyboard.reset()
                cmd_now.zero_()
                env.unwrapped.reset()
                obs = env.get_observations()

            # keyboard -> apply per-axis signs (KB_SIGN) -> clamp -> SLEW -> command buffer.
            # If any arrow feels reversed, flip that axis in KB_SIGN at the top of this file.
            kb_cmd = keyboard.advance() * kb_sign                # [3] on sim device
            cmd_target = torch.clamp(kb_cmd, -cmd_max, cmd_max)

            # heading-hold: cancel residual strafe yaw, but only while translating (a
            # correction at standstill makes it step in place chasing tiny heading errors)
            yaw = euler_xyz_from_quat(asset.data.root_quat_w)[2][0]
            translating = torch.abs(cmd_target[:2]).max() > MOVE_DEADZONE
            user_turning = torch.abs(cmd_target[2]) > TURN_DEADZONE
            if user_turning or not translating:
                target_yaw = yaw                                  # turning OR stopped -> re-latch, no hold
            elif heading_hold_on:
                yaw_err = wrap_to_pi(target_yaw - yaw)
                cmd_target[2] = torch.clamp(HEADING_KP * yaw_err, -cmd_max[2], cmd_max[2])

            max_step = slew_per_s * dt
            cmd_now += torch.clamp(cmd_target - cmd_now, -max_step, max_step)
            cmd_term.vel_command_b[:] = cmd_now

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if policy_has_reset:
                policy.reset(dones)

            # telemetry ~1x/s: commanded vs actual speed + the chosen gait frequency
            step_count += 1
            if step_count % 50 == 0:
                v = env.unwrapped.scene["robot"].data.root_lin_vel_b[0]
                print(f"[drive] cmd vx={cmd_now[0]:+.2f} vy={cmd_now[1]:+.2f} wz={cmd_now[2]:+.2f} | "
                      f"actual vx={v[0]:+.3f} vy={v[1]:+.3f} m/s | gait_freq={act_term.gait_freq[0]:.2f} Hz")

        # pace to real time — driving feel depends on it
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
