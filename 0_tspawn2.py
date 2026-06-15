# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn and test a custom Spider Robot articulation with Push Disturbance.

Usage:
    isaaclab.bat -p tspawn.py
"""

import argparse
import math
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Spawn custom Spider Robot and test push recovery.")
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
    # ใช้ dt=0.01 ตามเดิมของคุณ (หากต้องการฟิสิกส์ที่แม่นยำขึ้นในอนาคตแนะนำ 0.005 ครับ)
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

        # ทุกๆ 1000 เฟรม ทำการ Reset หุ่นกลับไปจุดเริ่มต้น
        if count % 1000 == 0:
            root_state = default_root_state.clone()
            joint_pos = default_joint_pos.clone()
            joint_vel = default_joint_vel.clone()

            spider.write_root_pose_to_sim(root_state[:, :7])
            spider.write_root_velocity_to_sim(root_state[:, 7:])
            spider.write_joint_state_to_sim(joint_pos, joint_vel)
            spider.reset()
            print("\n[INFO]: ---------- Robot reset to cfg init_state ----------")

        # =====================================================================
        # 🥊 ระบบจำลองการตบ/ผลักหุ่นยนต์ (Push Disturbance)
        # =====================================================================
        # ทุกๆ 300 เฟรม (ประมาณ 3 วินาที) จะทำการส่งแรงผลักแบบสุ่มในแนวแกน X และ Y
        if count % 300 == 150 and count % 1000 < 850: 
            # สร้างความเร็วลัพธ์จำลองจากการโดนกระแทก (Linear Velocity Offset ในหน่วย m/s)
            # สุ่มทิศทางและสุ่มแรงผลักติดลบหรือบวก ช่วงประมาณ -0.4 ถึง 0.4 m/s 
            push_vel_x = (torch.rand(spider.num_instances, device=spider.device) * 2.0 - 1.0) * 2.0
            push_vel_y = (torch.rand(spider.num_instances, device=spider.device) * 2.0 - 1.0) * 2.0
            
            # ดึงค่า Velocity ปัจจุบันของหุ่นออกมา
            current_root_vel = spider.data.root_vel_w.clone()
            
            # ใส่แรงผลักเข้าไปที่ความเร็วของ Base โดยตรง (แกน X และ Y)
            current_root_vel[:, 0] += push_vel_x
            current_root_vel[:, 1] += push_vel_y
            
            # เขียนค่าความเร็วใหม่กลับเข้าไปในระบบฟิสิกส์ทันที (เปรียบเสมือนหุ่นโดนตบจนปลิว)
            spider.write_root_velocity_to_sim(current_root_vel)
            
            print(f"Hey Nig i punch you DAMN X: {push_vel_x.item():.3f} m/s, Y: {push_vel_y.item():.3f} m/s")

        # ให้ actuator พยายามดึงข้อต่อรักษามุมท่ายืนตลอดเวลา
        spider.set_joint_position_target(default_joint_pos)

        spider.write_data_to_sim()
        sim.step()
        spider.update(sim.get_physics_dt())

        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()