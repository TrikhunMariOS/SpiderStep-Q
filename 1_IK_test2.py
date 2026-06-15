import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =================================================================
# 1. กำหนดโครงสร้างขาหุ่นยนต์ (เหมือนเดิม)
# =================================================================
L1 = 35.0  # Coxa
L2 = 65.0  # Femur
L3 = 100.0  # Tibia

def spider_IK(x, y, z):
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2) - L1
    D = np.sqrt(r**2 + z**2)
    if D > (L2 + L3) or D < np.abs(L2 - L3):
        raise ValueError("Out of workspace")
    cos_alpha = (L2**2 + D**2 - L3**2) / (2 * L2 * D)
    cos_beta  = (L2**2 + L3**2 - D**2) / (2 * L2 * L3)
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    beta  = np.arccos(np.clip(cos_beta, -1.0, 1.0))
    theta2 = np.arctan2(z, r) + alpha
    theta3 = np.pi - beta
    return theta1, theta2, theta3

def spider_FK(t1, t2, t3):
    p0 = np.array([0, 0, 0])
    p1 = np.array([L1 * np.cos(t1), L1 * np.sin(t1), 0])
    p2 = np.array([(L1 + L2 * np.cos(t2)) * np.cos(t1), (L1 + L2 * np.cos(t2)) * np.sin(t1), L2 * np.sin(t2)])
    p3 = np.array([
        (L1 + L2 * np.cos(t2) + L3 * np.cos(t2 - t3)) * np.cos(t1),
        (L1 + L2 * np.cos(t2) + L3 * np.cos(t2 - t3)) * np.sin(t1),
        L2 * np.sin(t2) + L3 * np.sin(t2 - t3)
    ])
    return p0, p1, p2, p3

# =================================================================
# 2. สร้างเส้นทางการก้าวขาแบบวงรี (Ellipse Trajectory Generator)
# =================================================================
num_points = 50
gait_trajectory = []

# กำหนดพารามิเตอร์การก้าว
stride_length = 40.0  # ความยาวก้าว (เดินหน้า-ถอยหลัง)
step_height = 20.0    # ความสูงตอนยกขาพ้นพื้น
center_x = 50.0       # ตำแหน่งกึ่งกลางขาในแนวแกน X
center_y = 30.0       # ตำแหน่งกึ่งกลางขาในแนวแกน Y
ground_z = -40.0      # ระดับพื้นดิน (แกน Z)

for i in range(num_points):
    phi = 2 * np.pi * i / num_points  # มุมเฟสการเดิน (0 ถึง 360 องศา)
    
    # คำนวณพิกัด X ขยับไป-กลับ
    x = center_x + (stride_length / 2.0) * np.cos(phi)
    y = center_y  # ให้เดินตรงไปข้างหน้าเฉยๆ แกน Y เลยคงที่
    
    # คำนวณพิกัด Z (ถ้าครึ่งลูปแรกให้ยกขา สูงตามฟังก์ชัน Sine, ครึ่งลูปหลังให้เหยียบพื้นลากตรง)
    if phi < np.pi:
        # Swing Phase: ยกขาขึ้นลอยกลางอากาศ
        z = ground_z + step_height * np.sin(phi)
    else:
        # Stance Phase: ขาเหยียบพื้นแน่น ลากเป็นเส้นตรงเพื่อดันตัว
        z = ground_z
        
    gait_trajectory.append((x, y, z))

# =================================================================
# 3. เตรียมการทำแอนิเมชันกราฟ 3 มิติ
# =================================================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# วาดเส้นทางวงรีสีเทาจางๆ ไว้ล่วงหน้าให้เห็นรอยทางเดิน
traj_np = np.array(gait_trajectory)
ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], '--', color='gray', alpha=0.6, label='Trajectory Path')

# สร้างตัวแปร Line Object ของขาและปลายเท้าที่จะขยับใน Animation
leg_line, = ax.plot([], [], [], '-o', color='blue', markersize=8, markerfacecolor='red', linewidth=3, label='Moving Leg')
foot_dot = ax.scatter([], [], [], color='green', marker='x', s=100)

ax.set_xlim(-20, 100)
ax.set_ylim(-20, 100)
ax.set_zlim(-80, 20)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Spiderbot Leg Trajectory Control Animation')
ax.legend()

# ฟังก์ชันอัปเดตเฟรมของแอนิเมชัน
def update(frame):
    global foot_dot
    # ดึงพิกัดเป้าหมายของเฟรมนั้นๆ มา
    x, y, z = gait_trajectory[frame]
    
    try:
        # 1. คำนวณ IK เพื่อแปลงพิกัดเป้าหมายเป็นองศามอเตอร์
        t1, t2, t3 = spider_IK(x, y, z)
        
        # 2. นำองศาไปเข้า FK เพื่อหาจุดพิกัดข้อต่อในการวาดกราฟ
        p0, p1, p2, p3 = spider_FK(t1, t2, t3)
        
        # อัปเดตพิกัดเส้นขาหุ่นยนต์
        leg_line.set_data([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]])
        leg_line.set_3d_properties([p0[2], p1[2], p2[2], p3[2]])
        
        # ลบจุดปลายเท้าเก่าและวาดจุดใหม่ทับพิกัดเป้าหมาย
        foot_dot.remove()
        foot_dot = ax.scatter(x, y, z, color='green', marker='O', s=50)
        
    except ValueError:
        pass
    return leg_line,

# สั่งรันแอนิเมชันวนลูปต่อเนื่อง
ani = animation.FuncAnimation(fig, update, frames=num_points, interval=50, blit=False)
plt.show()