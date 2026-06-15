# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn and test a custom Spider Robot articulation.

Usage:
    isaaclab.bat -p tspawn.py

Optional:
    isaaclab.bat -p tspawn.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Spawn custom Spider Robot with ArticulationCfg.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after app launch
# -----------------------------------------------------------------------------
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

# IMPORTANT:
# Put this file in the same folder as spider_robot_cfg.py,
# or change this import path to match where your cfg file is located.
from robot.spider_robot_cfg import SPIDER_CFG


def design_scene() -> Articulation:
    """Create ground, light, and spider robot articulation."""

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # Light
    light_cfg = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/lightDistant", light_cfg, translation=(1.0, 0.0, 10.0))

    # Robot articulation
    spider_cfg = SPIDER_CFG.replace(prim_path="/World/Spider")
    spider = Articulation(spider_cfg)

    return spider


def main():
    """Main function."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[0.7, 0.7, 0.45],
        target=[0.0, 0.0, 0.08],
    )

    spider = design_scene()

    sim.reset()

    print("[INFO]: Setup complete.")
    print(f"[INFO]: Number of joints: {spider.num_joints}")
    print(f"[INFO]: Joint names: {spider.joint_names}")
    print(f"[INFO]: Body names: {spider.body_names}")

    default_root_state = spider.data.default_root_state.clone()
    default_joint_pos = spider.data.default_joint_pos.clone()
    default_joint_vel = spider.data.default_joint_vel.clone()

    count = 0

    while simulation_app.is_running():

        if count % 300 == 0:
            root_state = default_root_state.clone()
            joint_pos = default_joint_pos.clone()
            joint_vel = default_joint_vel.clone()

            spider.write_root_pose_to_sim(root_state[:, :7])
            spider.write_root_velocity_to_sim(root_state[:, 7:])
            spider.write_joint_state_to_sim(joint_pos, joint_vel)
            spider.reset()

            print("[INFO]: Robot reset to cfg init_state.")

        # ให้ actuator พยายาม hold ท่า init_state ตลอดเวลา
        spider.set_joint_position_target(default_joint_pos)

        spider.write_data_to_sim()
        sim.step()
        spider.update(sim.get_physics_dt())

        count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()
