import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def parse_microtubule_data(filename):
    """解析微管坐标数据 - 修复版本"""
    data = {}
    current_pf = None
    current_coords = []

    print(f"正在解析文件: {filename}")

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # print(f"处理第{i}行: {line}")  # 调试信息

        if line.startswith('PF #'):
            # 保存前一个原纤维的数据
            if current_pf is not None and current_coords:
                data[current_pf] = np.array(current_coords)
                print(f"保存 {current_pf}: {len(current_coords)} 个坐标点")

            # 开始新的原纤维
            current_pf = line
            current_coords = []
            i += 2  # 跳过标题行
        elif line and current_pf is not None and not line.startswith('Q'):
            # 解析坐标数据，处理空字符串
            values = line.split(',')
            coords = []
            for val in values:
                val = val.strip()
                if val:  # 只处理非空字符串
                    try:
                        coords.append(float(val))
                    except ValueError as e:
                        print(f"警告: 无法转换 '{val}' 为浮点数，跳过该值")
                        continue

            if len(coords) == 6:  # Q1-Q6
                current_coords.append(coords)
            elif len(coords) > 0:
                print(f"警告: 第{i + 1}行有 {len(coords)} 个坐标值，期望6个")
            i += 1
        else:
            i += 1

    # 保存最后一个原纤维
    if current_pf is not None and current_coords:
        data[current_pf] = np.array(current_coords)
        print(f"保存 {current_pf}: {len(current_coords)} 个坐标点")

    print(f"解析完成，共找到 {len(data)} 条原纤维")
    return data


def parse_microtubule_data_pandas(filename):
    """使用pandas解析微管坐标数据 - 更稳定的方法"""
    print(f"使用pandas解析文件: {filename}")

    try:
        # 读取CSV文件
        df = pd.read_csv(filename, header=None)
        print(f"文件形状: {df.shape}")

        data = {}
        current_pf = None
        current_coords = []

        for index, row in df.iterrows():
            line = ' '.join(str(x) for x in row.values if pd.notna(x))

            if 'PF #' in line:
                # 保存前一个原纤维的数据
                if current_pf is not None and current_coords:
                    data[current_pf] = np.array(current_coords)

                # 开始新的原纤维
                current_pf = line.strip()
                current_coords = []
            elif current_pf is not None and not any(
                    x in str(line).upper() for x in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']):
                # 提取数字
                values = []
                for item in row.values:
                    if pd.notna(item):
                        try:
                            values.append(float(item))
                        except (ValueError, TypeError):
                            continue

                if len(values) >= 6:  # 至少需要6个坐标
                    current_coords.append(values[:6])

        # 保存最后一个原纤维
        if current_pf is not None and current_coords:
            data[current_pf] = np.array(current_coords)

        print(f"解析完成，共找到 {len(data)} 条原纤维")
        return data

    except Exception as e:
        print(f"pandas解析失败: {e}")
        # 回退到原始方法
        return parse_microtubule_data_simple(filename)


def parse_microtubule_data_simple(filename):
    """简化的解析方法"""
    print(f"使用简化方法解析: {filename}")

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    data = {}
    sections = content.split('PF #')

    for section in sections[1:]:  # 跳过第一个空元素
        lines = section.strip().split('\n')
        if not lines:
            continue

        pf_number = lines[0].split()[0] if lines[0] else "1"
        pf_name = f"PF #{pf_number}"
        coords = []

        for line in lines[2:]:  # 跳过PF标题和Q标题
            line = line.strip()
            if line:
                # 提取所有数字
                numbers = []
                for part in line.split(','):
                    part = part.strip()
                    if part:
                        try:
                            numbers.append(float(part))
                        except ValueError:
                            continue

                if len(numbers) >= 6:
                    coords.append(numbers[:6])

        if coords:
            data[pf_name] = np.array(coords)
            print(f"找到 {pf_name}: {len(coords)} 个坐标点")

    return data


def plot_3d_comparison(init_data, evolved_data):
    """3D对比可视化"""
    fig = plt.figure(figsize=(20, 8))

    # 初始结构
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, len(init_data)))

    for i, (pf, coords) in enumerate(init_data.items()):
        if len(coords) > 0:
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            ax1.scatter(x, y, z, c=[colors[i]], label=pf, s=50, alpha=0.7)
            ax1.plot(x, y, z, c=colors[i], alpha=0.5)

    ax1.set_title('初始结构 - init_coord.csv', fontsize=14)
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.set_zlabel('Z坐标')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 演化后结构
    ax2 = fig.add_subplot(122, projection='3d')

    for i, (pf, coords) in enumerate(evolved_data.items()):
        if len(coords) > 0:
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            ax2.scatter(x, y, z, c=[colors[i]], label=pf, s=50, alpha=0.7)
            ax2.plot(x, y, z, c=colors[i], alpha=0.5)

    ax2.set_title('演化后结构 - coord.csv', fontsize=14)
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    ax2.set_zlabel('Z坐标')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('microtubule_3d_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def quick_visualization(init_file, evolved_file):
    """快速可视化函数"""
    print("=== 微管结构可视化 ===")

    # 使用简化解析方法
    init_data = parse_microtubule_data_simple(init_file)
    evolved_data = parse_microtubule_data_simple(evolved_file)

    if not init_data or not evolved_data:
        print("错误: 无法解析数据文件")
        return

    print(f"初始数据: {len(init_data)} 条原纤维")
    print(f"演化数据: {len(evolved_data)} 条原纤维")

    # 创建可视化
    fig = plt.figure(figsize=(15, 10))

    # 1. 3D结构对比
    ax1 = fig.add_subplot(221, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, len(init_data)))

    for i, (pf, coords) in enumerate(init_data.items()):
        if len(coords) > 0:
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            ax1.scatter(x, y, z, c=[colors[i]], s=30, alpha=0.6)
            ax1.plot(x, y, z, c=colors[i], alpha=0.4, linewidth=1)

    ax1.set_title('初始结构')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(222, projection='3d')
    for i, (pf, coords) in enumerate(evolved_data.items()):
        if len(coords) > 0:
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            ax2.scatter(x, y, z, c=[colors[i]], s=30, alpha=0.6)
            ax2.plot(x, y, z, c=colors[i], alpha=0.4, linewidth=1)

    ax2.set_title('演化后结构')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # 2. 位移分析
    ax3 = fig.add_subplot(223)
    displacements = []

    for pf in init_data:
        if pf in evolved_data and len(init_data[pf]) > 0 and len(evolved_data[pf]) > 0:
            min_len = min(len(init_data[pf]), len(evolved_data[pf]))
            disp = np.linalg.norm(evolved_data[pf][:min_len] - init_data[pf][:min_len], axis=1)
            displacements.extend(disp)

    if displacements:
        ax3.hist(displacements, bins=20, alpha=0.7, color='skyblue')
        ax3.set_xlabel('位移量')
        ax3.set_ylabel('频次')
        ax3.set_title(f'位移分布 (平均: {np.mean(displacements):.3f})')
        ax3.grid(True, alpha=0.3)

    # 3. 径向投影
    ax4 = fig.add_subplot(224)
    for i, (pf, coords) in enumerate(init_data.items()):
        if len(coords) > 0:
            x, y = coords[:, 0], coords[:, 1]
            ax4.scatter(x, y, c=[colors[i]], s=20, alpha=0.6, label=pf)

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('XY平面投影')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('microtubule_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"初始结构蛋白亚基数: {sum(len(coords) for coords in init_data.values())}")
    print(f"演化结构蛋白亚基数: {sum(len(coords) for coords in evolved_data.values())}")
    if displacements:
        print(f"平均位移: {np.mean(displacements):.6f}")
        print(f"最大位移: {np.max(displacements):.6f}")
        print(f"位移标准差: {np.std(displacements):.6f}")


# 主执行函数
def main():
    # 文件路径 - 请根据实际情况修改
    init_file = 'E:\gits\MT_tubulin-master\experimental_results\exp1\init_coord.csv'
    evolved_file = 'E:\gits\MT_tubulin-master\experimental_results\exp1\coord.csv'

    # 执行快速可视化
    quick_visualization(init_file, evolved_file)


if __name__ == "__main__":
    main()

"""
=== 统计信息 ===
初始结构蛋白亚基数: 52
演化结构蛋白亚基数: 52
平均位移: 0.035352
最大位移: 0.078748
位移标准差: 0.027804
"""