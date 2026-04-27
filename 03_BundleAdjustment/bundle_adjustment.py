import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt # 引入绘图库
from pytorch3d.transforms import euler_angles_to_matrix

DATA_DIR = "data"
OUTPUT_DIR = "output"
VIS_DIR = os.path.join(OUTPUT_DIR, "vis_results")
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_W, IMG_H = 1024, 1024
CX, CY = IMG_W / 2.0, IMG_H / 2.0
NUM_VIEWS = 50 # 50个不同视角
NUM_POINTS = 20000 # 20000个3D点

print("Loading data...")
points2d_data = np.load(f"{DATA_DIR}/points2d.npz")
obs_list = []
for i in range(NUM_VIEWS):
    obs_list.append(points2d_data[f"view_{i:03d}"])
obs_2d = torch.from_numpy(np.stack(obs_list)).float().to(DEVICE) # (V, N, 3)

colors_rgb = np.load(f"{DATA_DIR}/points3d_colors.npy") # 用于渲染

# 相机内参
init_f = IMG_H / (2.0 * np.tan(np.deg2rad(60.0 / 2.0))) # 初始FoV建议
focal_length = nn.Parameter(torch.tensor(init_f, device=DEVICE))

# 相机外参
euler_angles = nn.Parameter(torch.zeros((NUM_VIEWS, 3), device=DEVICE)) # 初始Euler角为0
translations = nn.Parameter(torch.tensor([[0.0, 0.0, -3.0]], device=DEVICE).repeat(NUM_VIEWS, 1)) # 初始平移d=3

points3d = nn.Parameter(torch.randn((NUM_POINTS, 3), device=DEVICE) * 0.1) # 初始在原点附近

# Adam 优化器
optimizer = optim.Adam([focal_length, euler_angles, translations, points3d], lr=1e-2)

loss_history = [] 

def project_points(pts3d, f, eulers, trans):
    R = euler_angles_to_matrix(eulers, convention="XYZ") # Euler角参数化旋转
    # 相机变换
    cam_pts = torch.matmul(R.unsqueeze(1), pts3d.unsqueeze(2)).squeeze(-1) + trans.unsqueeze(1)
    
    xc, yc, zc = cam_pts[..., 0], cam_pts[..., 1], cam_pts[..., 2]
    
    # 投影公式
    # $$u = -f * Xc/Zc + cx, v = f * Yc/Zc + cy$$
    u = -f * (xc / zc) + CX
    v = f * (yc / zc) + CY
    return torch.stack([u, v], dim=-1)

print("Starting optimization...")
num_steps = 1000
for step in range(num_steps + 1):
    optimizer.zero_grad()
    
    pred_2d = project_points(points3d, focal_length, euler_angles, translations)
    
    # 最小化2D重投影误差
    observed_xy = obs_2d[..., :2]
    visibility = obs_2d[..., 2]
    
    # 计算误差平方和
    error = torch.sum((pred_2d - observed_xy)**2, dim=-1)
    loss = (error * visibility).sum() / (visibility.sum() + 1e-6)
    
    loss.backward()
    optimizer.step()
    
    # 记录 Loss
    loss_history.append(loss.item())
    
    if step % 100 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")


plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Re-projection Loss')
plt.yscale('log') 
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Bundle Adjustment Optimization Loss')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
print(f"Loss curve saved to {OUTPUT_DIR}/loss_curve.png")
# plt.show() 


final_pts = points3d.detach().cpu().numpy()
with open(f"{OUTPUT_DIR}/reconstructed.obj", 'w') as f_obj:
    for i in range(NUM_POINTS):
        # 格式：v x y z r g b
        f_obj.write(f"v {final_pts[i,0]} {final_pts[i,1]} {final_pts[i,2]} "
                   f"{colors_rgb[i,0]} {colors_rgb[i,1]} {colors_rgb[i,2]}\n")

# 依相机渲染点云投影图
print("Rendering final projections...")
with torch.no_grad():
    final_pred_2d = project_points(points3d, focal_length, euler_angles, translations).cpu().numpy()

colors_bgr = (colors_rgb[:, [2, 1, 0]] * 255).astype(np.uint8)

for i in [0, 12, 25, 37, 49]: # 选取特定视角进行可视化
    key = f"view_{i:03d}"
    img = cv2.imread(f"{DATA_DIR}/images/{key}.png")
    if img is None: continue
    
    pts_2d = final_pred_2d[i]
    vis = obs_2d[i, :, 2].cpu().numpy().astype(bool)
    
    for j in range(NUM_POINTS):
        if vis[j]:
            x, y = int(pts_2d[j, 0]), int(pts_2d[j, 1])
            if 0 <= x < IMG_W and 0 <= y < IMG_H:
                color = tuple(int(c) for c in colors_bgr[j])
                cv2.circle(img, (x, y), 2, color, -1)

    cv2.imwrite(f"{VIS_DIR}/{key}_reconstruction.png", img)
    print(f"Saved reconstruction overlay for {key}")

print("Done!")