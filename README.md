# SpiderStep-Q
### Reinforcement-Learning-Enhanced Locomotion for a Spider-Inspired Quadruped Robot
<img width="1051" height="398" alt="Screenshot 2026-06-12 022948 - Copy" src="https://github.com/user-attachments/assets/9b52b433-4b37-4b29-9790-a898028129bb" />


โครงงานนี้เป็นโครงงานที่ถูกพัฒนาขึ้นเพื่อศึกษาค้นคว้าเกี่ยวกับ Reinforcement Learning (RL) สำหรับควบคุมการเคลื่อนไหวหุ่นยนต์รูปแบบแมงมุม 4 ขา(3 DOF) โดยมีการใช้ gait เป็นพื้นฐาน

แทนที่จะให้ RL เรียนรู้การเดินตั้งแต่ศูนย์ โครงงานนี้ใช้แนวคิด **Residual RL** คือมี procedural gait (การเดินที่เขียนด้วยสูตรคณิตศาสตร์ + IK) เป็น baseline ที่ทำให้หุ่นยืนและเดินได้อยู่แล้ว จากนั้นให้ RL เรียนรู้แค่ส่วนเสริม(residual)เล็กๆเพื่อปรับแก้การทรงตัวและการเดินให้ดีขึ้น

ฝึกและทดสอบบน **Isaac Lab 2.x.x / Isaac Sim 5.1** ด้วย PPO (rsl_rl)

---

## คุณสมบัติหลัก

- หุ่นยนต์แมงมุม 4 ขา 12 DOF (ขาละ 3 ข้อ: Coxa / Femur / Tibia)
- Procedural gait แบบ Phoenix-style : feedforward CoG shift + Bezier swing trajectory + analytic IK
- Residual RL ที่เพิ่ม RL เป็นส่วนเสริมเล็กๆบน gait baseline 
- มีโหมดขับเองด้วยคีย์บอร์ด (`play_drive.py`) เพื่อทดสอบหุ่นกับอุปสรรค : ใช้ปุ่มลูกศร แทนการควบคุม WASD , Z/X เพื่อหมุนตัว , R เพื่อรีเซ็ต

---

## โครงสร้างโปรเจค

```text
MY_Final/
├── 0_tspawn*.py                     # ทดสอบ spawn หุ่นเข้า sim
├── 1_IK_*.py                        # ทดสอบ Inverse Kinematics ทีละขา / ทั้งตัว
├── 2_Trajectory_*.py                # ทดสอบ trajectory + FK verification
├── 3_gait_*.py / 4_gait_*.py        # ทดสอบ crawl gait / Bezier trajectory
├── 5_OrientationCompensation_test5.py  # procedural gait 
│
├── robot/
│   ├── spider_robot_cfg.py          # ArticulationCfg ของหุ่น
│   └── Robot_USD_v.1.02Final/       # โมเดลหุ่น: USD + URDF + STL meshes
│       └── SpiderBot_v1.02_Final.usd
│
└── rl/                              # ★ ระบบ Residual RL หลัก
    ├── __init__.py                  # ลงทะเบียน gym tasks
    ├── gait_torch.py                # SpiderGaitEngine — gait แบบ batched torch
    ├── gait_numpy_ref.py            # reference แบบ numpy (เก็บค่าคงที่ของ gait)
    ├── flat_env_cfg.py              # env บนพื้นราบ (ฝึกอันนี้ก่อน)
    ├── rough_env_cfg.py             # env พื้นขรุขระ
    ├── mdp/
    │   ├── gait_residual_action.py  # GaitResidualAction 
    │   ├── gait_foot_offset_action.py
    │   ├── observations.py          # gait_phase observation
    │   └── rewards.py
    ├── agents/rsl_rl_ppo_cfg.py     # hyperparameter ของ PPO
    ├── train_residual.py            # สคริปต์เริ่มฝึก
    ├── play_residual.py             # สคริปต์เล่น checkpoint
    ├── play_drive.py                # ขับหุ่นเองด้วยคีย์บอร์ด
    ├── play_gait_only.py            # ขับด้วย gait อย่างเดียว (residual = 0)
    └── test_*.py                    
```


---

## ความต้องการของระบบ (Requirements)

| รายการ | เวอร์ชัน |
|---|---|
| OS | Windows |
| Isaac Lab | 2.3.2 |
| Isaac Sim | 5.1 (Kit 107.3) |
| NumPy | `1.26.0` (Isaac Lab 2.3.2 pin `numpy<2`) |
| rsl-rl-lib | `5.0.1` |
| tensordict | `0.7.2` |
| h5py | `3.11.0` |
| GPU | รัน 4096 envs ได้สบายสำหรับหุ่นขนาดนี้ (~5 GB VRAM) |

> 🗿 **NumPy ต้องเป็น 1.26.0 เท่านั้น** ถ้าเผลออัปเป็น 2.x จะเกิด error `DLL load failed _multiarray_umath` แล้วแอป crash ตอนเปิด แก้ด้วย:
> ```bat
> python -m pip install --no-cache-dir "numpy==1.26.0"
> ```

> 🗿 เวอร์ชันของ `rsl-rl-lib`, `tensordict`, `h5py` ที่ Isaac Lab ลงมาให้เดิมใช้กับโปรเจคนี้ไม่ได้ ต้อง pin เป็นเวอร์ชันด้านบน (ดูคำสั่งติดตั้งในหัวข้อถัดไป)

---

## การติดตั้ง

1. ติดตั้ง Isaac Lab 2.3.2 + Isaac Sim 5.1 ตามคู่มือทางการ (แนะนำแบบ pip):
   https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

2. ติดตั้ง dependency ที่โปรเจคนี้ต้องการ (เวอร์ชันที่ Isaac Lab ลงมาให้เดิมใช้ไม่ได้)
   รันใน Python environment เดียวกับ Isaac Lab:

   เช็คว่า version ที่ถูกติดตั้งมาถูกมั้ย
   ```bat
   pip show rsl-rl-lib
   pip show tensordict
   pip show h5py
   ```
   ถ้าไม่ตรงให้
   ```bat
   pip uninstall rsl-rl-lib
   pip uninstall tensordict
   pip uninstall h5py
   ```
   ตามด้วยการติดตัั้ง lib ใหม่
   ```bat
   python -m pip install "rsl-rl-lib==5.0.1" "tensordict==0.7.2" "h5py==3.11.0"
   ```
   ถ้าเจอปัญหา cache/เวอร์ชันเก่าค้าง ให้บังคับลงใหม่:
   ```bat
   python -m pip install --no-cache-dir --force-reinstall "rsl-rl-lib==5.0.1" "tensordict==0.7.2" "h5py==3.11.0"
   ```

3. Clone โปรเจคนี้เข้าไปไว้ใต้โฟลเดอร์ `scripts` ของ Isaac Lab:
   ```bat
   cd C:\Users\<user>\IsaacLab\scripts
   git clone https://github.com/TrikhunMariOS/SpiderStep.git MY_Final
   ```

4. ไม่ต้องแก้ path ของ USD — `robot/spider_robot_cfg.py` ใช้ **relative path** หาไฟล์ USD จากตำแหน่งของตัวเองอยู่แล้ว ขอแค่โฟลเดอร์ `robot/Robot_USD_v.1.02Final/` อยู่ครบ (ห้ามลบ)

---
<img width="590" height="118" alt="Screenshot 2026-06-15 174613 - Copy" src="https://github.com/user-attachments/assets/995637aa-2d90-4d4e-b6f5-636d966faeb4" />

## วิธีรัน

รันทุกคำสั่งจากโฟลเดอร์ root ของ Isaac Lab ผ่าน `isaaclab.bat -p ...`

```bat
::  ตรวจสอบ IK (ไม่เปิดแอป) — ยืนยันว่า gait_torch ตรงกับ numpy ref
isaaclab.bat -p scripts/MY_Final/rl/test_ik_validation.py

::  gait อย่างเดียว (residual = 0) — baseline reference ที่ควรเดินได้
isaaclab.bat -p scripts/MY_Final/rl/play_gait_only.py --num_envs 16

::  ฝึกบนพื้นราบ (ฝึกอันนี้ก่อน)
isaaclab.bat -p scripts/MY_Final/rl/train_residual.py ^
--task Isaac-Velocity-Flat-Spider-Residual-v0 --num_envs 4096 --headless

::  เล่น checkpoint ที่ฝึกแล้ว
isaaclab.bat -p scripts/MY_Final/rl/play_residual.py ^
  --task Isaac-Velocity-Flat-Spider-Residual-Play-v0 --num_envs 16 ^
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_<n>.pt

:: ขับหุ่นเองด้วยคีย์บอร์ด (ต้องมี GUI, ห้าม --headless)
isaaclab.bat -p scripts/MY_Final/rl/play_drive.py ^
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_<n>.pt

:: ดูกราฟด้วย tensorboard
tensorboard --logdir <โฟลเดอร์ที่เก็บโมเดลเอาไว้>
```

Gym tasks ที่มี: `Isaac-Velocity-{Flat,Rough}-Spider-Residual-v0` (และ `-Play-v0`)

*หมายเหตุ* : มีโมเดลที่เทรนเสร็จแล้วอยู่ที่โฟลเดอร์ "Best_model_1800" โดยที่โมเดลเป็นไฟล์ model_1800.pt
---

## หมายเหตุสำคัญ (สำหรับผู้ที่จะพัฒนาต่อ)

- **EMA บน joint target ห้ามเอาออก** (`joint_smoothing_alpha = 0.12`) — ถ้าเอาออก swing ที่เร็วของ gait จะกระชากข้อต่อ underdamped จนหุ่นกระเด็น/ล้ม นี่คือบั๊กสำคัญที่สุดในโปรเจค
- **ใช้ `dt = 0.0025`, `decimation = 8`** — actuator (ImplicitActuatorCfg) ถูก tune มาที่ dt นี้ ถ้าเพิ่ม dt PD จะไม่เสถียร
- คำสั่งความเร็วต้อง ≤ ~0.12 m/s — เพราะ gait cap stride ไว้ที่ MAX_STRIDE = 100 mm ถ้าสั่งเร็วกว่านี้จะ track ไม่ได้
- ก่อนโทษ RL ให้ลองตั้ง residual_scale = 0 (gait อย่างเดียว) แล้วเทียบกับ play_gait_only เสมอ — baseline ต้องเดินได้ก่อน
---

## License

เผยแพร่ภายใต้ [MIT License](LICENSE) — © 2026 Trikhun MariOS
