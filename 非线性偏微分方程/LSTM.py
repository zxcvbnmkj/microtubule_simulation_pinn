import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

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
        # 输入序列
        x = self.coord_data[idx:idx + self.sequence_length]
        # 输出目标（下一个时间步）
        y = self.coord_data[idx + self.sequence_length:idx + self.sequence_length + self.predict_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM前向传播
        out, _ = self.lstm(x)
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


def load_and_preprocess_data(coord_file, init_coord_file):
    """加载和预处理微管坐标数据 - 修复版本"""
    print("开始加载微管坐标数据...")

    # 使用之前验证过的解析函数
    coord_data = parse_microtubule_data_simple(coord_file)
    init_coord_data = parse_microtubule_data_simple(init_coord_file)

    if not coord_data or not init_coord_data:
        raise ValueError("无法解析坐标数据文件，请检查文件格式")

    print(f"成功解析: 初始数据 {len(init_coord_data)} 条原纤维, 演化数据 {len(coord_data)} 条原纤维")

    # 提取所有坐标并合并
    all_coords = []

    # 按原纤维顺序处理数据
    for pf_num in range(1, 14):
        pf_name = f"PF #{pf_num}"

        if pf_name in init_coord_data:
            pf_data = init_coord_data[pf_name]
            all_coords.extend(pf_data)
            print(f"{pf_name} 初始: {len(pf_data)} 个亚基")

        if pf_name in coord_data:
            pf_data = coord_data[pf_name]
            all_coords.extend(pf_data)
            print(f"{pf_name} 演化: {len(pf_data)} 个亚基")

    # 转换为numpy数组
    combined_coords = np.array(all_coords)
    print(f"合并后数据形状: {combined_coords.shape}")

    if combined_coords.size == 0:
        raise ValueError("没有提取到有效坐标数据")

    # 标准化
    scaler = StandardScaler()
    normalized_coords = scaler.fit_transform(combined_coords)
    print(f"标准化后数据形状: {normalized_coords.shape}")

    return normalized_coords, scaler


def train_simple_model(model, dataloader, criterion, optimizer, device):
    """训练普通LSTM模型"""
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # 调整target形状以匹配output
        target_reshaped = target.reshape(target.shape[0], -1)
        loss = criterion(output, target_reshaped)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_simple_model(model, dataloader, criterion, device):
    """验证普通LSTM模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 调整target形状以匹配output
            target_reshaped = target.reshape(target.shape[0], -1)
            loss = criterion(output, target_reshaped)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def predict_and_visualize(model, test_loader, device, scaler, num_samples=5):
    """预测并可视化结果 - 修复版本"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break

            data, target = data.to(device), target.to(device)
            output = model(data)

            # 调整target形状以匹配output
            target_reshaped = target.reshape(target.shape[0], -1)

            # 转换为numpy并反标准化
            output_np = output.cpu().numpy()  # shape: (batch_size, output_features)
            target_np = target_reshaped.cpu().numpy()  # shape: (batch_size, output_features)

            # 反标准化 - 确保是2D数组
            output_denorm = scaler.inverse_transform(output_np)
            target_denorm = scaler.inverse_transform(target_np)

            predictions.append(output_denorm)
            actuals.append(target_denorm)

    # 合并所有批次的数据
    all_predictions = np.vstack(predictions)
    all_actuals = np.vstack(actuals)

    print(f"预测数据形状: {all_predictions.shape}")
    print(f"实际数据形状: {all_actuals.shape}")

    # 可视化预测结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # 选择前6个特征进行可视化
    feature_names = ['Q1(X)', 'Q2(Y)', 'Q3(Z)', 'Q4(角度1)', 'Q5(角度2)', 'Q6(角度3)']

    # 随机选择一些样本点进行可视化
    sample_indices = np.random.choice(min(20, len(all_predictions)), size=min(10, len(all_predictions)), replace=False)

    for i in range(min(6, all_predictions.shape[1])):
        pred_vals = all_predictions[sample_indices, i]
        actual_vals = all_actuals[sample_indices, i]

        x_pos = np.arange(len(sample_indices))
        width = 0.35

        axes[i].bar(x_pos - width / 2, actual_vals, width, label='实际值', alpha=0.7)
        axes[i].bar(x_pos + width / 2, pred_vals, width, label='预测值', alpha=0.7)
        axes[i].set_title(f'{feature_names[i]} 预测对比')
        axes[i].set_xlabel('样本索引')
        axes[i].set_ylabel('数值')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # 设置x轴刻度
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels([f'S{idx}' for idx in sample_indices])

    plt.tight_layout()
    plt.savefig('lstm_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return all_predictions, all_actuals


def analyze_prediction_accuracy(predictions, actuals):
    """分析预测精度"""
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)

    print(f"\n=== 预测精度分析 ===")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")

    # 计算每个特征的误差
    feature_names = ['Q1(X)', 'Q2(Y)', 'Q3(Z)', 'Q4(角度1)', 'Q5(角度2)', 'Q6(角度3)']
    for i in range(min(6, predictions.shape[1])):
        feature_mse = np.mean((predictions[:, i] - actuals[:, i]) ** 2)
        feature_mae = np.mean(np.abs(predictions[:, i] - actuals[:, i]))
        print(f"{feature_names[i]} - MSE: {feature_mse:.6f}, MAE: {feature_mae:.6f}")


def debug_data_shapes(model, test_loader, device):
    """调试数据形状"""
    print("\n=== 数据形状调试 ===")
    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= 1:  # 只看第一个批次
                break

            print(f"输入数据形状: {data.shape}")  # 应该是 (batch, seq_len, features)
            print(f"目标数据形状: {target.shape}")  # 应该是 (batch, predict_len, features)

            data, target = data.to(device), target.to(device)
            output = model(data)
            print(f"模型输出形状: {output.shape}")

            # 调整target形状
            target_reshaped = target.reshape(target.shape[0], -1)
            print(f"调整后目标形状: {target_reshaped.shape}")


if __name__ == '__main__':
    # 配置参数
    config = {
        'sequence_length': 5,  # 输入序列长度
        'predict_length': 1,  # 预测长度
        'batch_size': 8,  # 批大小
        'hidden_size': 128,  # LSTM隐藏层大小
        'num_layers': 2,  # LSTM层数
        'learning_rate': 0.001,  # 学习率
        'epochs': 100,  # 训练轮数
        'dropout': 0.2  # Dropout率
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        # 加载数据
        print("加载数据...")
        coords_data, scaler = load_and_preprocess_data(
            'E:\\gits\\MT_tubulin-master\\experimental_results\\exp1\\coord.csv',
            'E:\\gits\\MT_tubulin-master\\experimental_results\\exp1\\init_coord.csv'
        )

        # 创建数据集
        dataset = MicrotubuleDataset(
            coords_data,
            sequence_length=config['sequence_length'],
            predict_length=config['predict_length']
        )

        print(f"数据集大小: {len(dataset)}")

        # 分割数据集
        train_size = int(0.7 * len(dataset))  # 70% 训练
        val_size = int(0.15 * len(dataset))  # 15% 验证
        test_size = len(dataset) - train_size - val_size  # 15% 测试

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        # 创建普通LSTM模型
        input_size = coords_data.shape[1]  # 6个坐标(Q1-Q6)
        output_size = input_size * config['predict_length']

        print(f"输入特征数: {input_size}, 输出特征数: {output_size}")

        model = SimpleLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size,
            dropout=config['dropout']
        )

        model = model.to(device)

        # 调试数据形状
        debug_data_shapes(model, test_loader, device)

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # 训练循环
        print("\n开始训练普通LSTM模型...")
        train_losses = []
        val_losses = []

        for epoch in range(config['epochs']):
            train_loss = train_simple_model(model, train_loader, criterion, optimizer, device)
            val_loss = validate_simple_model(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{config["epochs"]}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 绘制训练曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('lstm_training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 在测试集上预测并可视化
        print("\n在测试集上进行预测...")
        predictions, actuals = predict_and_visualize(model, test_loader, device, scaler)

        # 分析预测精度
        analyze_prediction_accuracy(predictions, actuals)

        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'config': config
        }, 'simple_lstm_microtubule_model.pth')

        print("训练完成！模型已保存为 'simple_lstm_microtubule_model.pth'")

    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback

        traceback.print_exc()