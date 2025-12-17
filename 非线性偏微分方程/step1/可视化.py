import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 读取数据
df = pd.read_csv(r'E:\pinn_mt\MT_tubulin-master\results_6k_100.csv')

# 筛选数据：layer=3，纤维丝=0（即第一个蛋白，Q11-Q16）
df_filtered = df[df['layer'] == 3].copy()

# 提取第一个蛋白的坐标数据
fiber_1_data = df_filtered[['step', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16']].copy()

# 重命名列以便更好理解
fiber_1_data = fiber_1_data.rename(columns={
    'Q11': 'X', 'Q12': 'Y', 'Q13': 'Z',
    'Q14': 'phi', 'Q15': 'theta', 'Q16': 'psi'
})

# 创建双Y轴图
fig, ax1 = plt.subplots(figsize=(26, 8))

# 设置颜色
xyz_colors = ['blue', 'green', 'red']  # XYZ坐标颜色
euler_colors = ['orange', 'purple', 'brown']  # 欧拉角颜色

# 左侧Y轴：XYZ坐标
ax1.set_xlabel('时间步 (step)', fontsize=14, fontweight='bold')
ax1.set_ylabel('XYZ坐标值', fontsize=14, fontweight='bold', color='blue')


initial_values = fiber_1_data.iloc[0]
fiber_1_data['X'] = fiber_1_data['X'] - initial_values['X']
fiber_1_data['Y'] = fiber_1_data['Y'] - initial_values['Y']
fiber_1_data['Z'] = fiber_1_data['Z'] - initial_values['Z']

# 绘制XYZ坐标线
x_line, = ax1.plot(fiber_1_data['step']-12, fiber_1_data['X'],
                   color=xyz_colors[0], linewidth=2, markersize=4, label='X')
y_line, = ax1.plot(fiber_1_data['step'], fiber_1_data['Y'],
                   color=xyz_colors[1], linewidth=2, markersize=4, label='Y')
z_line, = ax1.plot(fiber_1_data['step']-14, fiber_1_data['Z'],
                   color=xyz_colors[2], linewidth=2, markersize=4, label='Z')

ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3, linestyle='--')

# 右侧Y轴：欧拉角
ax2 = ax1.twinx()
ax2.set_ylabel('欧拉角 (弧度)', fontsize=14, fontweight='bold', color='orange')

# 绘制欧拉角线
phi_line, = ax2.plot(fiber_1_data['step'], fiber_1_data['phi'],
                     color=euler_colors[0], linewidth=2, markersize=4, label='φ (phi)')
theta_line, = ax2.plot(fiber_1_data['step'], fiber_1_data['theta'],
                       color=euler_colors[1], linewidth=2, markersize=4, label='θ (theta)')
psi_line, = ax2.plot(fiber_1_data['step'], fiber_1_data['psi'],
                     color=euler_colors[2], linewidth=2, markersize=4, label='ψ (psi)')

ax2.tick_params(axis='y', labelcolor='orange')

# 设置X轴刻度为整数
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# 添加标题
plt.title(f'Layer 3, 纤维丝 0 的蛋白坐标和欧拉角变化\n'
          f'时间步范围: {fiber_1_data["step"].min()} - {fiber_1_data["step"].max()}',
          fontsize=16, fontweight='bold', pad=20)

# 合并图例
lines = [x_line, y_line, z_line, phi_line, theta_line, psi_line]
labels = [line.get_label() for line in lines]

# 将图例放在图外
ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1),
           fontsize=12, frameon=True, shadow=True, fancybox=True)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()