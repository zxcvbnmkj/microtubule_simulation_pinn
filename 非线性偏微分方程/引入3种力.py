import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
from mpl_toolkits.mplot3d import Axes3D

from 可视化 import parse_microtubule_data_simple

warnings.filterwarnings('ignore')


class MicrotubuleDataset(Dataset):
    def __init__(self, coord_data, sequence_length=10, predict_length=1):
        self.coord_data = coord_data
        self.sequence_length = sequence_length
        self.predict_length = predict_length

    def __len__(self):
        return len(self.coord_data) - self.sequence_length - self.predict_length + 1

    def __getitem__(self, idx):
        x = self.coord_data[idx:idx + self.sequence_length]
        y = self.coord_data[idx + self.sequence_length:idx + self.sequence_length + self.predict_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class PhysicsInformedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(PhysicsInformedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        # 物理参数 - 可学习的力常数
        self.stretch_coeff = nn.Parameter(torch.tensor(1.0))  # 拉伸系数 S
        self.bend_coeff = nn.Parameter(torch.tensor(0.1))  # 弯曲系数 B
        self.twist_coeff = nn.Parameter(torch.tensor(0.05))  # 扭曲系数 K

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class MicrotubulePhysics:
    """微管物理力学模型"""

    def __init__(self, stretch_coeff=1.0, bend_coeff=0.1, twist_coeff=0.05):
        self.S = stretch_coeff  # 拉伸刚度
        self.B = bend_coeff  # 弯曲刚度
        self.K = twist_coeff  # 扭曲刚度

    def calculate_stretch_force(self, current_pos, neighbor_pos, equilibrium_dist=1.0):
        """计算拉伸力"""
        displacement = current_pos - neighbor_pos
        distance = torch.norm(displacement, dim=-1, keepdim=True)
        stretch = distance - equilibrium_dist
        force = -self.S * stretch * (displacement / (distance + 1e-8))
        return force

    def calculate_bend_force(self, pos1, pos2, pos3):
        """计算弯曲力 - 三个连续点的曲率"""
        vec1 = pos2 - pos1
        vec2 = pos3 - pos2

        # 计算曲率
        cross_prod = torch.cross(vec1, vec2)
        cross_norm = torch.norm(cross_prod, dim=-1, keepdim=True)
        vec1_norm = torch.norm(vec1, dim=-1, keepdim=True)
        vec2_norm = torch.norm(vec2, dim=-1, keepdim=True)

        curvature = cross_norm / (vec1_norm * vec2_norm + 1e-8)
        bend_force = -self.B * curvature.unsqueeze(-1) * (pos3 - pos1)
        return bend_force

    def calculate_twist_force(self, current_angles, neighbor_angles, equilibrium_twist=0.0):
        """计算扭曲力 - 基于角度差异"""
        angle_diff = current_angles - neighbor_angles - equilibrium_twist
        twist_force = -self.K * angle_diff
        return twist_force

    def calculate_total_force(self, positions, angles):
        """计算总作用力"""
        batch_size, seq_len, _ = positions.shape

        total_forces = torch.zeros_like(positions)
        total_torques = torch.zeros_like(angles)

        for i in range(seq_len):
            # 拉伸力 - 与相邻亚基的相互作用
            if i > 0:  # 与前一个亚基
                stretch_force = self.calculate_stretch_force(
                    positions[:, i], positions[:, i - 1])
                total_forces[:, i] += stretch_force.squeeze(-1)

            if i < seq_len - 1:  # 与后一个亚基
                stretch_force = self.calculate_stretch_force(
                    positions[:, i], positions[:, i + 1])
                total_forces[:, i] += stretch_force.squeeze(-1)

            # 弯曲力 - 需要三个连续点
            if i > 0 and i < seq_len - 1:
                bend_force = self.calculate_bend_force(
                    positions[:, i - 1], positions[:, i], positions[:, i + 1])
                total_forces[:, i] += bend_force.squeeze(-1)

            # 扭曲力 - 角度相互作用
            if i > 0:
                twist_torque = self.calculate_twist_force(
                    angles[:, i], angles[:, i - 1])
                total_torques[:, i] += twist_torque

            if i < seq_len - 1:
                twist_torque = self.calculate_twist_force(
                    angles[:, i], angles[:, i + 1])
                total_torques[:, i] += twist_torque

        return total_forces, total_torques


class EnhancedPhysicsLoss(nn.Module):
    """增强的物理约束损失函数 - 修复版本"""

    def __init__(self, alpha=0.1, beta=0.01, gamma=0.01):
        super(EnhancedPhysicsLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.physics_model = MicrotubulePhysics()

    def forward(self, predictions, targets, previous_states):
        # 1. 基础均方误差损失
        mse_loss = nn.MSELoss()(predictions, targets)

        # 2. 布朗运动约束
        displacements = predictions - previous_states
        displacement_norms = torch.norm(displacements, dim=-1)
        brownian_constraint = torch.abs(torch.var(displacement_norms) - 1.0)

        # 3. 力学平衡约束 - 只在有足够样本时计算
        batch_size = predictions.shape[0]
        if batch_size > 1:
            # 分离位置和角度信息
            pred_positions = predictions[:, :3].unsqueeze(1)  # 添加序列维度
            pred_angles = predictions[:, 3:].unsqueeze(1)

            # 计算物理力
            forces, torques = self.physics_model.calculate_total_force(
                pred_positions, pred_angles)

            # 力应该与位移相关
            expected_displacement = forces.squeeze(1)
            actual_displacement = displacements[:, :3]

            # 只在维度匹配时计算损失
            if expected_displacement.shape == actual_displacement.shape:
                force_balance_loss = nn.MSELoss()(actual_displacement, expected_displacement)
            else:
                force_balance_loss = torch.tensor(0.0)
        else:
            force_balance_loss = torch.tensor(0.0)

        total_loss = (mse_loss +
                      self.alpha * brownian_constraint +
                      self.beta * force_balance_loss)

        return total_loss

class MicrotubuleVisualizer:
    """微管可视化类"""

    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))

    def plot_3d_structure(self, coordinates, title="Microtubule 3D Structure"):
        """绘制3D微管结构"""
        ax = self.fig.add_subplot(231, projection='3d')

        # 提取位置坐标 (Q1-Q3)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]

        ax.scatter(x, y, z, c=z, cmap='viridis', s=20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

    def plot_force_distribution(self, forces, title="Force Distribution"):
        """绘制力分布"""
        ax = self.fig.add_subplot(232)

        force_magnitudes = np.linalg.norm(forces, axis=1)
        ax.hist(force_magnitudes, bins=20, alpha=0.7)
        ax.set_xlabel('Force Magnitude')
        ax.set_ylabel('Frequency')
        ax.set_title(title)

    def plot_energy_evolution(self, energies, title="Energy Evolution"):
        """绘制能量演化"""
        ax = self.fig.add_subplot(233)
        ax.plot(energies)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Energy')
        ax.set_title(title)

    def plot_coordinate_evolution(self, coordinates, coord_idx, title="Coordinate Evolution"):
        """绘制坐标演化"""
        ax = self.fig.add_subplot(234)
        ax.plot(coordinates[:, coord_idx])
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'Q{coord_idx + 1}')
        ax.set_title(f'{title} - Q{coord_idx + 1}')

    def plot_physical_parameters(self, stretch_params, bend_params, twist_params):
        """绘制物理参数"""
        ax = self.fig.add_subplot(235)

        epochs = range(len(stretch_params))
        ax.plot(epochs, stretch_params, label='Stretch Coeff', marker='o')
        ax.plot(epochs, bend_params, label='Bend Coeff', marker='s')
        ax.plot(epochs, twist_params, label='Twist Coeff', marker='^')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Physical Parameters Evolution')
        ax.legend()
        ax.grid(True)

    def save_visualization(self, filename="microtubule_analysis.png"):
        """保存可视化结果"""
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()


def load_and_preprocess_data(coord_file, init_coord_file):
    """加载和预处理微管坐标数据"""
    print("开始加载微管坐标数据...")

    coord_data = parse_microtubule_data_simple(coord_file)
    init_coord_data = parse_microtubule_data_simple(init_coord_file)

    if not coord_data or not init_coord_data:
        raise ValueError("无法解析坐标数据文件，请检查文件格式")

    print(f"成功解析: 初始数据 {len(init_coord_data)} 条原纤维, 演化数据 {len(coord_data)} 条原纤维")

    all_coords = []

    for pf_num in range(1, 14):
        pf_name = f"PF #{pf_num}"

        if pf_name in init_coord_data:
            pf_data = init_coord_data[pf_name]
            all_coords.extend(pf_data)

        if pf_name in coord_data:
            pf_data = coord_data[pf_name]
            all_coords.extend(pf_data)

    combined_coords = np.array(all_coords)
    print(f"合并后数据形状: {combined_coords.shape}")

    if combined_coords.size == 0:
        raise ValueError("没有提取到有效坐标数据")

    scaler = StandardScaler()
    normalized_coords = scaler.fit_transform(combined_coords)
    print(f"标准化后数据形状: {normalized_coords.shape}")

    return normalized_coords, scaler


def train_model(model, dataloader, criterion, optimizer, device, physics_criterion=None):
    """训练模型 - 修复版本"""
    model.train()
    total_loss = 0

    stretch_params = []
    bend_params = []
    twist_params = []

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # 确保数据维度正确
        batch_size, seq_len, features = data.shape
        target = target.reshape(batch_size, -1)  # 重塑目标为2D

        optimizer.zero_grad()
        output = model(data)

        if physics_criterion is not None:
            # 使用最后时间步的数据作为previous_states
            previous_states = data[:, -1, :]
            loss = physics_criterion(output, target, previous_states)
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 记录物理参数
        if hasattr(model, 'stretch_coeff'):
            stretch_params.append(model.stretch_coeff.item())
            bend_params.append(model.bend_coeff.item())
            twist_params.append(model.twist_coeff.item())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, stretch_params, bend_params, twist_params


def validate_model(model, dataloader, criterion, device, physics_criterion=None):
    """验证模型 - 修复版本"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            batch_size, seq_len, features = data.shape
            target = target.reshape(batch_size, -1)  # 重塑目标为2D

            output = model(data)

            if physics_criterion is not None:
                previous_states = data[:, -1, :]
                loss = physics_criterion(output, target, previous_states)
            else:
                loss = criterion(output, target)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def analyze_physical_properties(model, dataloader, device, scaler):
    """分析物理性质 - 修复版本"""
    model.eval()
    all_predictions = []
    all_targets = []

    physics_model = MicrotubulePhysics()

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 重塑数据为2D (batch_size * sequence_length, features)
            batch_size = output.shape[0]

            # 将预测和目标重塑为2D
            output_2d = output.reshape(-1, output.shape[-1])  # (batch_size * predict_length, features)
            target_2d = target.reshape(-1, target.shape[-1])  # (batch_size * predict_length, features)

            # 反标准化
            output_denorm = scaler.inverse_transform(output_2d.cpu().numpy())
            target_denorm = scaler.inverse_transform(target_2d.cpu().numpy())

            all_predictions.append(output_denorm)
            all_targets.append(target_denorm)

    # 合并所有批次
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")

    # 计算物理力 - 只使用前几个样本避免内存问题
    sample_size = min(100, len(predictions))
    pred_positions = torch.FloatTensor(predictions[:sample_size, :3]).unsqueeze(1)
    pred_angles = torch.FloatTensor(predictions[:sample_size, 3:]).unsqueeze(1)

    forces, torques = physics_model.calculate_total_force(pred_positions, pred_angles)

    return predictions, targets, forces.squeeze().numpy(), torques.squeeze().numpy()


def plot_3d_structure(self, coordinates, title="Microtubule 3D Structure"):
    """绘制3D微管结构 - 修复版本"""
    ax = self.fig.add_subplot(231, projection='3d')

    # 如果坐标太多，随机采样一部分显示
    if len(coordinates) > 1000:
        indices = np.random.choice(len(coordinates), 1000, replace=False)
        coords_sample = coordinates[indices]
    else:
        coords_sample = coordinates

    # 提取位置坐标 (Q1-Q3)
    x = coords_sample[:, 0]
    y = coords_sample[:, 1]
    z = coords_sample[:, 2]

    ax.scatter(x, y, z, c=z, cmap='viridis', s=10, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

if __name__ == '__main__':
    # 配置参数
    config = {
        'sequence_length': 5,
        'predict_length': 1,
        'batch_size': 8,
        'hidden_size': 128,
        'num_layers': 2,
        'learning_rate': 0.001,
        'epochs': 100,
        'model_type': 'lstm',
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    coords_data, scaler = load_and_preprocess_data(
        'E:\gits\MT_tubulin-master\experimental_results\exp1\coord.csv',
        'E:\gits\MT_tubulin-master\experimental_results\exp1\init_coord.csv'
    )

    # 创建数据集
    dataset = MicrotubuleDataset(coords_data,
                                 sequence_length=config['sequence_length'],
                                 predict_length=config['predict_length'])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 创建模型
    input_size = coords_data.shape[1]
    output_size = input_size * config['predict_length']

    if config['model_type'] == 'lstm':
        model = PhysicsInformedLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size
        )
    else:
        # 可以添加Transformer版本
        model = PhysicsInformedLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size
        )

    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    physics_criterion = EnhancedPhysicsLoss(alpha=0.1, beta=0.01, gamma=0.01)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 训练循环
    print("开始训练...")
    train_losses = []
    val_losses = []

    # 记录物理参数
    all_stretch_params = []
    all_bend_params = []
    all_twist_params = []

    for epoch in range(config['epochs']):
        train_loss, stretch_params, bend_params, twist_params = train_model(
            model, train_loader, criterion, optimizer, device, physics_criterion)
        val_loss = validate_model(model, val_loader, criterion, device, physics_criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 记录物理参数
        if stretch_params:
            all_stretch_params.extend(stretch_params)
            all_bend_params.extend(bend_params)
            all_twist_params.extend(twist_params)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            if hasattr(model, 'stretch_coeff'):
                print(f'Physical Params - S: {model.stretch_coeff.item():.4f}, '
                      f'B: {model.bend_coeff.item():.4f}, K: {model.twist_coeff.item():.4f}')

    # 分析物理性质
    print("分析物理性质...")
    predictions, targets, forces, torques = analyze_physical_properties(model, val_loader, device, scaler)

    # 可视化结果
    print("生成可视化...")
    visualizer = MicrotubuleVisualizer()

    # 绘制3D结构
    visualizer.plot_3d_structure(predictions, "Predicted Microtubule Structure")

    # 绘制力分布
    visualizer.plot_force_distribution(forces, "Force Distribution")

    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

    # 绘制物理参数演化
    if all_stretch_params:
        visualizer.plot_physical_parameters(all_stretch_params, all_bend_params, all_twist_params)

    # 保存综合可视化
    visualizer.save_visualization("microtubule_physics_analysis.png")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'config': config,
        'physical_params': {
            'stretch_coeff': model.stretch_coeff.item() if hasattr(model, 'stretch_coeff') else 1.0,
            'bend_coeff': model.bend_coeff.item() if hasattr(model, 'bend_coeff') else 0.1,
            'twist_coeff': model.twist_coeff.item() if hasattr(model, 'twist_coeff') else 0.05
        }
    }, 'microtubule_physics_model.pth')

    print("训练完成！模型和可视化结果已保存。")