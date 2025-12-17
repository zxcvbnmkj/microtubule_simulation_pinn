import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 临时解决OpenMP冲突问题

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 禁用可能导致冲突的其他后端
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端


# 生成虚拟数据 - 模拟单个蛋白在微管上的运动
def generate_microtubule_data(t_steps=1000):
    """生成微管蛋白运动的虚拟数据"""
    t = np.linspace(0, 10, t_steps)

    # 6个坐标：3个空间坐标 + 3个旋转/构象坐标
    # 1. x, y, z 空间位置 - 沿着微管轴向运动
    x = 0.1 * t + 0.05 * np.sin(2 * np.pi * 0.5 * t)  # 沿着微管的轴向运动
    y = 0.02 * np.sin(2 * np.pi * 0.3 * t)  # 横向摆动
    z = 0.01 * np.cos(2 * np.pi * 0.4 * t)  # 垂直方向波动

    # 2. 旋转角度：俯仰(pitch)，偏航(yaw)，滚转(roll)
    pitch = 0.1 * np.sin(2 * np.pi * 0.2 * t)  # 俯仰角
    yaw = 0.15 * np.cos(2 * np.pi * 0.25 * t)  # 偏航角
    roll = 0.05 * np.sin(2 * np.pi * 0.35 * t + np.pi / 4)  # 滚转角

    # 3. 构象状态（假设值在0-1之间变化）
    conformation = 0.5 + 0.3 * np.sin(2 * np.pi * 0.1 * t)

    # 添加一些噪声使数据更真实
    noise_level = 0.01
    x += np.random.normal(0, noise_level, t_steps)
    y += np.random.normal(0, noise_level / 2, t_steps)
    z += np.random.normal(0, noise_level / 2, t_steps)

    coordinates = np.vstack([x, y, z, pitch, yaw, roll]).T

    return t, coordinates, conformation


# 生成数据
t_steps = 1000
t, coordinates, conformation = generate_microtubule_data(t_steps)

# 创建美观的曲线图
fig = plt.figure(figsize=(18, 14))

# 使用GridSpec创建复杂的布局
gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)

# 设置颜色主题
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#5F0F40']

# 1. 主轨迹图
ax1 = fig.add_subplot(gs[0:2, 0:2])
# 绘制所有6个坐标的时间序列
for i in range(6):
    # 标准化以便在同一图中显示
    norm_data = (coordinates[:, i] - np.min(coordinates[:, i])) / \
                (np.max(coordinates[:, i]) - np.min(coordinates[:, i]))
    ax1.plot(t, norm_data + i, label=f'Coord {i + 1}',
             color=colors[i], linewidth=2, alpha=0.8)

ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Normalized Value', fontsize=12)
ax1.set_title('All 6 Coordinates Evolution', fontsize=14, fontweight='bold', pad=20)
ax1.legend(['X Position', 'Y Position', 'Z Position',
            'Pitch', 'Yaw', 'Roll'],
           loc='upper right', fontsize=10, ncol=2)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim([0, 10])

# 2-7. 各个坐标的详细视图
coord_names = [
    'X Position (μm)',
    'Y Position (μm)',
    'Z Position (μm)',
    'Pitch Angle (rad)',
    'Yaw Angle (rad)',
    'Roll Angle (rad)'
]

for i in range(6):
    row = i // 3 + 2
    col = i % 3
    ax = fig.add_subplot(gs[row, col])

    # 绘制坐标曲线
    ax.plot(t, coordinates[:, i], color=colors[i], linewidth=2.5, alpha=0.9,
            label=coord_names[i].split(' ')[0])

    # 添加填充区域表示变化范围
    ax.fill_between(t,
                    coordinates[:, i] - np.std(coordinates[:, i]) / 3,
                    coordinates[:, i] + np.std(coordinates[:, i]) / 3,
                    color=colors[i], alpha=0.15)

    # 统计信息
    mean_val = np.mean(coordinates[:, i])
    std_val = np.std(coordinates[:, i])
    min_val = np.min(coordinates[:, i])
    max_val = np.max(coordinates[:, i])

    # 添加统计线
    ax.axhline(y=mean_val, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axhline(y=mean_val + std_val, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(y=mean_val - std_val, color='gray', linestyle='--', alpha=0.4, linewidth=1)

    # 统计信息文本框
    stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nRange: [{min_val:.3f}, {max_val:.3f}]'
    ax.text(0.02, 0.95, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(coord_names[i], fontsize=11)
    ax.set_title(f'Coordinate {i + 1}: {coord_names[i].split(" ")[0]}',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim([0, 10])

    # 如果i是前3个（空间坐标），设置相同的Y轴范围以便比较
    if i < 3:
        ax.set_ylim([-0.1, 0.1])

# 8. 构象状态和能量图
ax8 = fig.add_subplot(gs[0, 2:])
# 计算动能（基于速度）
velocity_x = np.gradient(coordinates[:, 0], t)
kinetic_energy = 0.5 * velocity_x ** 2  # 简化的动能

ax8.plot(t, conformation, 'purple', linewidth=2.5, label='Conformation', alpha=0.8)
ax8.set_xlabel('Time (s)', fontsize=11)
ax8.set_ylabel('Conformation', fontsize=11, color='purple')
ax8.tick_params(axis='y', labelcolor='purple')
ax8.set_ylim([0, 1])

# 创建第二个Y轴用于动能
ax8_energy = ax8.twinx()
ax8_energy.plot(t, kinetic_energy, 'darkorange', linewidth=2,
                label='Kinetic Energy', alpha=0.7, linestyle='--')
ax8_energy.set_ylabel('Kinetic Energy (a.u.)', fontsize=11, color='darkorange')
ax8_energy.tick_params(axis='y', labelcolor='darkorange')

ax8.set_title('Conformation State and Energy', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.2)

# 合并图例
lines1, labels1 = ax8.get_legend_handles_labels()
lines2, labels2 = ax8_energy.get_legend_handles_labels()
ax8.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

# 9. 3D投影视图
ax9 = fig.add_subplot(gs[1, 2:])
# 创建热力图显示运动密度
from scipy.ndimage import gaussian_filter

# 创建2D直方图
heatmap, xedges, yedges = np.histogram2d(
    coordinates[:, 0], coordinates[:, 1], bins=50
)
heatmap = gaussian_filter(heatmap, sigma=1.5)

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax9.imshow(heatmap.T, extent=extent, origin='lower',
                cmap='viridis', aspect='auto', alpha=0.8)

# 叠加轨迹线
ax9.plot(coordinates[:, 0], coordinates[:, 1], 'white', linewidth=1, alpha=0.6)
ax9.scatter(coordinates[::50, 0], coordinates[::50, 1],
            c=t[::50], cmap='plasma', s=30, edgecolors='white', alpha=0.7)

ax9.set_xlabel('X Position (μm)', fontsize=11)
ax9.set_ylabel('Y Position (μm)', fontsize=11)
ax9.set_title('Protein Motion Density (X-Y Plane)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax9, label='Density')

# 10. 速度分析
ax10 = fig.add_subplot(gs[2:, 3])
# 计算所有方向的速度
velocities = np.gradient(coordinates[:, :3], t, axis=0)
speed = np.linalg.norm(velocities, axis=1)

# 速度分布直方图
ax10.hist(speed, bins=30, color='teal', alpha=0.7, edgecolor='black', density=True)

# 添加正态分布拟合
from scipy.stats import norm

mu, std = norm.fit(speed)
xmin, xmax = ax10.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
ax10.plot(x, p, 'r', linewidth=2, label=f'Fit: μ={mu:.3f}, σ={std:.3f}')

ax10.set_xlabel('Speed (μm/s)', fontsize=11)
ax10.set_ylabel('Probability Density', fontsize=11)
ax10.set_title('Speed Distribution', fontsize=12, fontweight='bold')
ax10.legend(fontsize=9)
ax10.grid(True, alpha=0.2)

# 11. 相关性矩阵（简化版）
ax11 = fig.add_subplot(gs[3, 0:3])
# 计算移动相关性
window_size = 100
correlation_over_time = []

for i in range(0, len(t) - window_size, 10):
    window_data = coordinates[i:i + window_size]
    corr_matrix = np.corrcoef(window_data.T)
    # 取X与其他坐标的相关性
    correlation_over_time.append(corr_matrix[0, 1:])

correlation_over_time = np.array(correlation_over_time)
time_midpoints = t[window_size // 2:-window_size // 2:10]

# 绘制相关性随时间变化
for i in range(5):
    ax11.plot(time_midpoints, correlation_over_time[:, i],
              label=f'X with Coord {i + 2}', linewidth=2, alpha=0.8)

ax11.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax11.set_xlabel('Time (s)', fontsize=11)
ax11.set_ylabel('Correlation Coefficient', fontsize=11)
ax11.set_title('Time-dependent Correlations with X Position',
               fontsize=12, fontweight='bold')
ax11.legend(loc='upper right', fontsize=9)
ax11.grid(True, alpha=0.2)
ax11.set_ylim([-1, 1])

# 添加总标题
plt.suptitle('Microtubule Protein Dynamics: 6-Dimensional Motion Analysis\n'
             'Time Evolution of Spatial and Angular Coordinates',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# 保存图像
plt.savefig('microtubule_protein_dynamics_advanced.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("=" * 60)
print("微管蛋白动力学仿真完成！")
print("=" * 60)
print(f"时间步数: {t_steps}")
print(f"时间范围: 0 到 {t[-1]:.1f} 秒")
print(f"总模拟时间: {t[-1]:.1f} 秒")
print(f"数据维度: {coordinates.shape}")
print("\n坐标统计信息:")
print("-" * 40)
for i, name in enumerate(['X', 'Y', 'Z', 'Pitch', 'Yaw', 'Roll']):
    print(f"{name}: mean={np.mean(coordinates[:, i]):.4f}, "
          f"std={np.std(coordinates[:, i]):.4f}, "
          f"range=[{np.min(coordinates[:, i]):.4f}, {np.max(coordinates[:, i]):.4f}]")
print("\n图像已保存为 'microtubule_protein_dynamics_advanced.png'")
print("=" * 60)