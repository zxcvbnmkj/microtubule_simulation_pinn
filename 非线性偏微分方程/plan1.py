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


class PhysicsInformedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(PhysicsInformedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM前向传播
        out, (hn, cn) = self.lstm(x)
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


class TransformerMicrotubule(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super(TransformerMicrotubule, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, x):
        # 输入投影和位置编码
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Transformer编码器
        x = self.transformer_encoder(x)

        # 取最后一个时间步并投影到输出维度
        x = self.output_projection(x[:, -1, :])
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class PhysicsLoss(nn.Module):
    """物理约束损失函数"""

    def __init__(self, alpha=0.1):
        super(PhysicsLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets, previous_states):
        # 1. 均方误差损失
        mse_loss = nn.MSELoss()(predictions, targets)

        # 2. 布朗运动约束：位移方差应该合理
        displacements = predictions - previous_states
        displacement_norms = torch.norm(displacements, dim=-1)

        # 爱因斯坦关系约束（简化版）
        brownian_constraint = torch.abs(torch.var(displacement_norms) - 1.0)  # 假设单位时间方差为1

        # 3. 连续性约束：相邻时间步变化不应太大
        continuity_loss = torch.mean(torch.abs(predictions[:, 1:] - predictions[:, :-1]))

        total_loss = mse_loss + self.alpha * brownian_constraint + 0.01 * continuity_loss
        return total_loss


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

def extract_protofilament_data(df, pf_number):
    """提取单个原纤维的数据"""
    pf_data = []
    current_line = 0

    while current_line < len(df):
        line = df.iloc[current_line]
        if f'PF #{pf_number}' in str(line.values):
            # 跳过标题行
            current_line += 2  # 跳过PF标题和Q1-Q6标题
            # 读取4个蛋白亚基的数据
            for i in range(4):
                if current_line < len(df):
                    coords = df.iloc[current_line].values
                    if len(coords) == 6:  # Q1-Q6
                        pf_data.append(coords)
                    current_line += 1
        else:
            current_line += 1

    return np.array(pf_data)


def train_model(model, dataloader, criterion, optimizer, device, physics_criterion=None):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        if physics_criterion is not None:
            # 使用物理约束损失
            loss = physics_criterion(output, target, data[:, -1, :])
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_model(model, dataloader, criterion, device, physics_criterion=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            if physics_criterion is not None:
                loss = physics_criterion(output, target, data[:, -1, :])
            else:
                loss = criterion(output, target)

            total_loss += loss.item()

    return total_loss / len(dataloader)


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
        'model_type': 'transformer',  # 'lstm' or 'transformer'
    }

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    coords_data, scaler = load_and_preprocess_data('E:\gits\MT_tubulin-master\experimental_results\exp1\coord.csv', 'E:\gits\MT_tubulin-master\experimental_results\exp1\init_coord.csv')

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
    input_size = coords_data.shape[1]  # 6个坐标(Q1-Q6)
    output_size = input_size * config['predict_length']

    if config['model_type'] == 'lstm':
        model = PhysicsInformedLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=output_size
        )
    else:  # transformer
        model = TransformerMicrotubule(
            input_size=input_size,
            d_model=128,
            nhead=8,
            num_layers=3,
            output_size=output_size
        )

    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    physics_criterion = PhysicsLoss(alpha=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 训练循环
    print("开始训练...")
    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        train_loss = train_model(model, train_loader, criterion, optimizer, device, physics_criterion)
        val_loss = validate_model(model, val_loader, criterion, device, physics_criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'config': config
    }, 'microtubule_model.pth')

    print("训练完成！模型已保存。")


def predict_future(model, initial_sequence, steps, device, scaler):
    """预测未来多个时间步"""
    model.eval()
    predictions = []
    current_sequence = initial_sequence.clone()

    with torch.no_grad():
        for _ in range(steps):
            # 预测下一个时间步
            pred = model(current_sequence.unsqueeze(0).to(device))
            pred = pred.cpu()

            # 更新序列
            new_sequence = torch.cat([current_sequence[1:], pred.reshape(1, -1)], dim=0)
            current_sequence = new_sequence

            # 反标准化并保存
            pred_denormalized = scaler.inverse_transform(pred.numpy().reshape(1, -1))
            predictions.append(pred_denormalized[0])

    return np.array(predictions)