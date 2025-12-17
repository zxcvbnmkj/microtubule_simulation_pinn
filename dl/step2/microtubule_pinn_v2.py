# -*- coding: utf-8 -*-
"""
微管动力学PINN仿真系统 V2
输入：当前微管坐标 q(t)
输出：下一时刻微管坐标 q(t+1)
物理约束：Langevin方程

架构：
1. 图神经网络（GNN）
2. MSE损失函数（数据拟合）
3. PINN物理约束（Langevin方程）
4. 总损失 = MSE损失 + λ × 物理损失
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 物理参数（与C++代码保持一致）====================
dc = 1e-10  # coordinate difference step
dp = 1e-2   # angle difference step
dt = 2.0 * 1e-12  # 时间步长 (秒)

# 模型参数
mt_rad = 25 / 2 * 1e-9  # 微管半径
k_B = 1.38 * 1e-23  # 玻尔兹曼常数 (J/K)
T = 300.0  # 温度 (K)

# 势能函数参数
r_0 = 0.12 * 1e-9
d = 0.25 * 1e-9

# 相互作用参数
A_la = 16.9 * k_B * T
b_la = 9.1 * k_B * T
A_lo = 17.6 * k_B * T
b_lo = 15.5 * k_B * T
k = 517 * k_B * T * 1e18  # inter-dimer bond strength

# 圆周率
pi = np.pi

# 流体粘度
h = 0.2  # Pa*s

# 蛋白单体半径
R = 2.0 * 1e-9  # m

# 摩擦系数向量 ξ
xi = torch.tensor([
    dt / (6 * pi * R * h),
    dt / (6 * pi * R * h),
    dt / (6 * pi * R * h),
    dt / (8 * pi * R**3 * h),
    dt / (8 * pi * R**3 * h),
    dt / (8 * pi * R**3 * h)
], device=device)


# ==================== 图结构定义 ====================
def build_microtubule_graph(n_filaments=13, n_subunits=4):
    """
    构建微管图结构
    节点：每个蛋白亚基 (总共 n_filaments × n_subunits = 52个节点)
    边：纵向连接（原纤维内相邻亚基）+ 横向连接（相邻原纤维的对应亚基）
    
    返回: edge_index (2, num_edges) - 边的索引
    """
    num_nodes = n_filaments * n_subunits
    edges = []
    
    # 节点编号: node_id = filament_id * n_subunits + subunit_id
    # filament_id: 0-12, subunit_id: 0-3
    
    # 1. 纵向边（原纤维内相邻亚基）
    for i in range(n_filaments):
        for j in range(n_subunits - 1):
            node_from = i * n_subunits + j
            node_to = i * n_subunits + j + 1
            edges.append([node_from, node_to])
            edges.append([node_to, node_from])  # 无向图，双向连接
    
    # 2. 横向边（相邻原纤维的对应亚基）
    for i in range(n_filaments):
        for j in range(n_subunits):
            # 左邻居原纤维
            i_left = (i - 1) % n_filaments
            node_from = i * n_subunits + j
            node_to = i_left * n_subunits + j
            edges.append([node_from, node_to])
            edges.append([node_to, node_from])  # 无向图
            
            # 右邻居原纤维
            i_right = (i + 1) % n_filaments
            node_to = i_right * n_subunits + j
            edges.append([node_from, node_to])
            edges.append([node_to, node_from])  # 无向图
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


# ==================== 图神经网络模型 ====================
class MicrotubuleDynamicsModel(nn.Module):
    """
    基于图神经网络的微管动力学模型
    输入: q(t) - 当前时刻所有蛋白亚基的坐标 (batch_size, 13, 4, 6)
    输出: q(t+1) - 下一时刻所有蛋白亚基的坐标 (batch_size, 13, 4, 6)
    
    图结构：
    - 节点：每个蛋白亚基（52个节点）
    - 边：纵向连接（原纤维内）+ 横向连接（原纤维间）
    """
    
    def __init__(self, n_filaments=13, n_subunits=4, hidden_size=128, num_layers=3):
        super(MicrotubuleDynamicsModel, self).__init__()
        self.n_filaments = n_filaments
        self.n_subunits = n_subunits
        self.num_nodes = n_filaments * n_subunits
        self.node_feature_dim = 6
        self.hidden_size = hidden_size
        
        # 构建图结构（边索引）
        self.register_buffer('edge_index', build_microtubule_graph(n_filaments, n_subunits))
        
        # 节点特征编码层
        self.input_encoder = nn.Linear(self.node_feature_dim, hidden_size)
        
        # GCN层
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.node_feature_dim)
        )
    
    def forward(self, q_current):
        """
        前向传播
        q_current: 当前坐标 (batch_size, 13, 4, 6) 或 (batch_size, 312)
        返回: 下一时刻坐标 (batch_size, 13, 4, 6)
        """
        batch_size = q_current.shape[0]
        
        # 转换为节点特征格式
        if len(q_current.shape) == 2:
            # (batch_size, 312) -> (batch_size, 13, 4, 6)
            q_current = q_current.view(batch_size, self.n_filaments, self.n_subunits, self.node_feature_dim)
        
        # 重塑为节点特征: (batch_size, num_nodes, node_feature_dim)
        node_features = q_current.view(batch_size, self.num_nodes, self.node_feature_dim)
        
        # 处理批次中的每个样本
        batch_outputs = []
        for b in range(batch_size):
            x = node_features[b]  # (num_nodes, node_feature_dim)
            
            # 输入编码
            x = self.input_encoder(x)  # (num_nodes, hidden_size)
            x = torch.relu(x)
            
            # GNN层（带残差连接）
            for gnn_layer in self.gnn_layers:
                x_new = torch.relu(gnn_layer(x, self.edge_index))
                x = x + x_new  # 残差连接
            
            # 输出解码
            x = self.output_decoder(x)  # (num_nodes, node_feature_dim)
            
            batch_outputs.append(x)
        
        # 堆叠批次
        output = torch.stack(batch_outputs, dim=0)  # (batch_size, num_nodes, node_feature_dim)
        
        # 重塑回 (batch_size, n_filaments, n_subunits, 6)
        output = output.view(batch_size, self.n_filaments, self.n_subunits, self.node_feature_dim)
        
        return output


# ==================== 物理约束计算 ====================
class LangevinPhysicsConstraint:
    """
    Langevin方程物理约束
    q^(m) = q^(m-1) - (dt/ξ_i) F^(m-1) + sqrt(2 k_B T (dt/ξ_i)) × N(0,1)
    """
    
    def __init__(self, n_filaments=13, n_subunits=4):
        self.n_filaments = n_filaments
        self.n_subunits = n_subunits
    
    def compute_forces(self, coords):
        """
        计算系统性力 F = -∇U
        coords: (batch_size, n_filaments, n_subunits, 6)
        返回: forces (batch_size, n_filaments, n_subunits, 6)
        """
        batch_size = coords.shape[0]
        forces = torch.zeros_like(coords)
        
        for b in range(batch_size):
            for i in range(self.n_filaments):
                for j in range(self.n_subunits):
                    coord = coords[b, i, j]
                    
                    # 纵向力（与原纤维内相邻亚基的相互作用）
                    if j > 0:
                        coord_prev = coords[b, i, j-1]
                        dr = coord[:3] - coord_prev[:3]
                        dist = torch.norm(dr)
                        if dist > 1e-10:
                            # LI势能的梯度: F = -k * (r - r0) * (r/r)
                            force_long = -k * (dist - 2*R) * (dr / dist)
                            forces[b, i, j, :3] += force_long
                    
                    if j < self.n_subunits - 1:
                        coord_next = coords[b, i, j+1]
                        dr = coord[:3] - coord_next[:3]
                        dist = torch.norm(dr)
                        if dist > 1e-10:
                            force_long = -k * (dist - 2*R) * (dr / dist)
                            forces[b, i, j, :3] += force_long
                    
                    # 横向力（与相邻原纤维的相互作用）
                    i_left = (i - 1) % self.n_filaments
                    i_right = (i + 1) % self.n_filaments
                    
                    if i_left != i:
                        coord_left = coords[b, i_left, j]
                        dr_lat = coord[:3] - coord_left[:3]
                        dist_lat = torch.norm(dr_lat)
                        if dist_lat > 1e-10:
                            # LA势能的梯度（简化）
                            force_lat = -A_la * (dr_lat / dist_lat) * torch.exp(-dist_lat / r_0)
                            forces[b, i, j, :3] += force_lat
                    
                    if i_right != i:
                        coord_right = coords[b, i_right, j]
                        dr_lat = coord[:3] - coord_right[:3]
                        dist_lat = torch.norm(dr_lat)
                        if dist_lat > 1e-10:
                            force_lat = -A_la * (dr_lat / dist_lat) * torch.exp(-dist_lat / r_0)
                            forces[b, i, j, :3] += force_lat
        
        return forces
    
    def compute_physics_residual(self, q_current, q_next_pred):
        """
        计算Langevin方程的物理残差
        q_next_pred 应该满足: q_next = q_current - (dt/ξ) * F + noise
        
        在训练时忽略随机项，只考虑确定性部分
        """
        # 计算力 F(q_current)
        forces = self.compute_forces(q_current)
        
        # Langevin方程的确定性部分
        xi_expanded = xi.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 6)
        deterministic = q_current - (dt / xi_expanded) * forces
        
        # 物理残差：预测值应该等于确定性部分
        physics_residual = q_next_pred - deterministic
        
        return physics_residual


# ==================== PINN损失函数 ====================
class PINNLoss(nn.Module):
    """
    PINN总损失函数
    总损失 = MSE损失 + λ × 物理损失
    """
    
    def __init__(self, physics_constraint, lambda_physics=0.1):
        super(PINNLoss, self).__init__()
        self.physics_constraint = physics_constraint
        self.lambda_physics = lambda_physics
        self.mse_loss = nn.MSELoss()
    
    def forward(self, q_next_pred, q_next_true, q_current):
        """
        计算总损失
        q_next_pred: 模型预测的下一时刻坐标
        q_next_true: 真实的下一时刻坐标（用于MSE）
        q_current: 当前时刻坐标（用于物理约束）
        """
        # 1. MSE损失（数据拟合损失）
        mse_loss_val = self.mse_loss(q_next_pred, q_next_true)
        
        # 2. 物理损失（Langevin方程约束）
        physics_residual = self.physics_constraint.compute_physics_residual(q_current, q_next_pred)
        physics_loss_val = torch.mean(physics_residual**2)
        
        # 3. 总损失
        total_loss = mse_loss_val + self.lambda_physics * physics_loss_val
        
        return total_loss, mse_loss_val, physics_loss_val


# ==================== 初始化微管结构 ====================
def initialize_microtubule(n_filaments=13, n_subunits=4):
    """初始化微管坐标"""
    init_coords = torch.zeros(n_filaments, n_subunits, 6, device=device)
    
    for i in range(n_filaments):
        for j in range(n_subunits):
            init_coords[i, j] = torch.tensor([
                mt_rad * np.cos(i * 2. * pi / 13.),  # x
                mt_rad * np.sin(i * 2. * pi / 13.),  # y
                (2. * j + 1.) * R + i * 12.0 / 13.0 * 1e-9,  # z
                0, 0, 0  # 角度
            ], device=device)
    
    return init_coords


# ==================== 生成训练数据 ====================
def generate_training_data(init_coords, n_steps=None, use_cpp_data=True, cpp_data_path=None):
    """
    从C++结果文件加载训练数据
    CSV格式: step,layer,Q11,Q12,Q13,Q14,Q15,Q16,Q21,...,Q136
    每个时间步有4行（layer 0-3），需要组合成完整的微管坐标 (13, 4, 6)
    """
    if cpp_data_path is None:
        # 默认路径
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cpp_data_path = os.path.join(base_dir, 'MT_tubulin-master', 'results_5k.csv')
    
    if not os.path.exists(cpp_data_path):
        raise FileNotFoundError("找不到C++结果文件: {}".format(cpp_data_path))
    
    df = pd.read_csv(cpp_data_path)
    unique_steps = sorted(df['step'].unique())
    
    if n_steps is not None:
        unique_steps = unique_steps[:n_steps+1]
    
    all_coords = []
    n_filaments = 13
    n_subunits = 4
    n_dims = 6
    
    # 遍历每个时间步
    for step in unique_steps:
        # 获取该时间步的所有layer数据
        step_data = df[df['step'] == step].sort_values('layer')
        
        if len(step_data) != n_subunits:
            continue
        
        # 初始化该时间步的坐标数组 (13, 4, 6)
        step_coords = torch.zeros(n_filaments, n_subunits, n_dims, device=device)
        
        # 遍历每个layer
        for layer_idx, (_, row) in enumerate(step_data.iterrows()):
            # 遍历每条原纤维（Q11-Q136）
            for pf_idx in range(n_filaments):
                # 提取该原纤维的6个坐标值
                q_values = []
                for dim_idx in range(n_dims):
                    col_name = 'Q{}{}'.format(pf_idx + 1, dim_idx + 1)
                    if col_name in row:
                        value = row[col_name]
                        if pd.notna(value):
                            q_values.append(float(value))
                        else:
                            q_values.append(0.0)
                    else:
                        q_values.append(0.0)
                
                # 前3个是位置坐标（nm），需要转换为米
                # 后3个是角度（弧度），保持不变
                step_coords[pf_idx, layer_idx, 0] = q_values[0] * 1e-9  # x: nm -> m
                step_coords[pf_idx, layer_idx, 1] = q_values[1] * 1e-9  # y: nm -> m
                step_coords[pf_idx, layer_idx, 2] = q_values[2] * 1e-9  # z: nm -> m
                step_coords[pf_idx, layer_idx, 3] = q_values[3]  # α: 弧度
                step_coords[pf_idx, layer_idx, 4] = q_values[4]  # β: 弧度
                step_coords[pf_idx, layer_idx, 5] = q_values[5]  # γ: 弧度
        
        all_coords.append(step_coords)
    
    if len(all_coords) == 0:
        raise ValueError("未能从CSV文件中提取到有效数据")
    
    all_coords_tensor = torch.stack(all_coords)
    return all_coords_tensor


# ==================== 数据集划分 ====================
def split_dataset(all_coords, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    划分数据集为训练集、验证集和测试集
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    n_samples = len(all_coords) - 1  # 减去1因为需要成对数据
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # 按时间顺序划分（保持时间连续性）
    train_indices = torch.arange(0, n_train)
    val_indices = torch.arange(n_train, n_train + n_val)
    test_indices = torch.arange(n_train + n_val, n_samples)
    
    return train_indices, val_indices, test_indices


# ==================== 评价指标 ====================
class EvaluationMetrics:
    """评价指标计算"""
    
    def __init__(self, n_filaments=13, n_subunits=4):
        self.n_filaments = n_filaments
        self.n_subunits = n_subunits
    
    def compute_metrics(self, q_pred, q_true, q_current=None, physics_constraint=None):
        """
        计算评价指标
        q_pred: 预测坐标 (batch_size, n_filaments, n_subunits, 6)
        q_true: 真实坐标 (batch_size, n_filaments, n_subunits, 6)
        q_current: 当前坐标（用于物理约束）
        """
        metrics = {}
        
        # 1. MSE (均方误差)
        mse = torch.mean((q_pred - q_true) ** 2)
        metrics['MSE'] = mse.item()
        
        # 2. MAE (平均绝对误差)
        mae = torch.mean(torch.abs(q_pred - q_true))
        metrics['MAE'] = mae.item()
        
        # 3. 位置误差（前3维）
        pos_pred = q_pred[:, :, :, :3]
        pos_true = q_true[:, :, :, :3]
        pos_mse = torch.mean((pos_pred - pos_true) ** 2)
        pos_mae = torch.mean(torch.abs(pos_pred - pos_true))
        metrics['Position_MSE'] = pos_mse.item()
        metrics['Position_MAE'] = pos_mae.item()
        metrics['Position_RMSE'] = torch.sqrt(pos_mse).item()
        
        # 4. 角度误差（后3维）
        angle_pred = q_pred[:, :, :, 3:]
        angle_true = q_true[:, :, :, 3:]
        angle_mse = torch.mean((angle_pred - angle_true) ** 2)
        angle_mae = torch.mean(torch.abs(angle_pred - angle_true))
        metrics['Angle_MSE'] = angle_mse.item()
        metrics['Angle_MAE'] = angle_mae.item()
        metrics['Angle_RMSE'] = torch.sqrt(angle_mse).item()
        
        # 5. 相对误差（百分比）
        relative_error = torch.mean(torch.abs(q_pred - q_true) / (torch.abs(q_true) + 1e-10))
        metrics['Relative_Error'] = relative_error.item() * 100  # 转换为百分比
        
        # 6. 物理约束误差（如果提供了physics_constraint）
        if physics_constraint is not None and q_current is not None:
            physics_residual = physics_constraint.compute_physics_residual(q_current, q_pred)
            physics_error = torch.mean(physics_residual ** 2)
            metrics['Physics_Constraint_Error'] = physics_error.item()
        
        # 7. R² 决定系数
        ss_res = torch.sum((q_true - q_pred) ** 2)
        ss_tot = torch.sum((q_true - torch.mean(q_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        metrics['R2'] = r2.item()
        
        return metrics
    
    def print_metrics(self, metrics, prefix=""):
        """打印评价指标"""
        print("\n{}: MSE={:.6e}, R²={:.4f}, 位置RMSE={:.6e}m, 角度RMSE={:.6e}rad".format(
            prefix, metrics['MSE'], metrics['R2'], 
            metrics['Position_RMSE'], metrics['Angle_RMSE']))
        if 'Physics_Constraint_Error' in metrics:
            print("  物理约束误差: {:.6e}".format(metrics['Physics_Constraint_Error']))


# ==================== 验证函数 ====================
def evaluate(model, data_indices, all_coords, batch_size=32, physics_constraint=None):
    """
    在验证集或测试集上评估模型
    """
    model.eval()
    metrics_calculator = EvaluationMetrics(model.n_filaments, model.n_subunits)
    
    all_preds = []
    all_trues = []
    all_currents = []
    
    with torch.no_grad():
        for batch_start in range(0, len(data_indices), batch_size):
            batch_indices = data_indices[batch_start:batch_start+batch_size]
            
            q_current_batch = all_coords[batch_indices]
            q_next_true_batch = all_coords[batch_indices + 1]
            
            # 模型预测
            q_next_pred_batch = model(q_current_batch)
            
            all_preds.append(q_next_pred_batch)
            all_trues.append(q_next_true_batch)
            all_currents.append(q_current_batch)
    
    # 合并所有批次
    q_pred_all = torch.cat(all_preds, dim=0)
    q_true_all = torch.cat(all_trues, dim=0)
    q_current_all = torch.cat(all_currents, dim=0)
    
    # 计算指标
    metrics = metrics_calculator.compute_metrics(
        q_pred_all, q_true_all, q_current_all, physics_constraint
    )
    
    return metrics, q_pred_all, q_true_all


# ==================== 训练函数 ====================
def train_pinn(model, init_coords, train_indices, val_indices, n_epochs=1000, 
               batch_size=32, lr=1e-3, lambda_physics=0.1, save_best=True):
    """
    训练PINN模型，包含验证集评估和最佳模型保存
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    physics_constraint = LangevinPhysicsConstraint(model.n_filaments, model.n_subunits)
    loss_fn = PINNLoss(physics_constraint, lambda_physics=lambda_physics)
    
    # 加载数据
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    cpp_data_path = os.path.join(project_root, 'MT_tubulin-master', 'results_5k.csv')
    
    if not os.path.exists(cpp_data_path):
        cpp_data_path = os.path.join('..', '..', 'MT_tubulin-master', 'results_5k.csv')
    
    all_coords = generate_training_data(init_coords, n_steps=None, use_cpp_data=True, cpp_data_path=cpp_data_path)
    all_coords = all_coords.to(device)
    
    # 训练历史
    train_losses = []
    train_mse_losses = []
    train_physics_losses = []
    val_losses = []
    val_metrics_history = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(n_epochs):
        # ========== 训练阶段 ==========
        model.train()
        epoch_total_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_physics_loss = 0.0
        n_batches = 0
        
        # 随机打乱训练数据
        train_shuffled = train_indices[torch.randperm(len(train_indices))]
        
        for batch_start in range(0, len(train_shuffled), batch_size):
            batch_indices = train_shuffled[batch_start:batch_start+batch_size]
            
            q_current_batch = all_coords[batch_indices]
            q_next_true_batch = all_coords[batch_indices + 1]
            
            optimizer.zero_grad()
            
            q_next_pred_batch = model(q_current_batch)
            
            total_loss, mse_loss_val, physics_loss_val = loss_fn(
                q_next_pred_batch, q_next_true_batch, q_current_batch
            )
            
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_mse_loss += mse_loss_val.item()
            epoch_physics_loss += physics_loss_val.item()
            n_batches += 1
        
        avg_train_loss = epoch_total_loss / n_batches
        avg_train_mse = epoch_mse_loss / n_batches
        avg_train_physics = epoch_physics_loss / n_batches
        
        train_losses.append(avg_train_loss)
        train_mse_losses.append(avg_train_mse)
        train_physics_losses.append(avg_train_physics)
        
        # ========== 验证阶段 ==========
        if len(val_indices) > 0:
            val_metrics, _, _ = evaluate(model, val_indices, all_coords, batch_size, physics_constraint)
            val_loss = val_metrics['MSE']  # 使用MSE作为验证损失
            val_losses.append(val_loss)
            val_metrics_history.append(val_metrics)
            
            # 保存最佳模型
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_path = "microtubule_pinn_v2_best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, best_model_path)
        
        if (epoch + 1) % 100 == 0:
            if len(val_indices) > 0:
                print("Epoch {}/{}, Train Loss: {:.6e}, Val MSE: {:.6e}, Val R²: {:.4f}".format(
                    epoch+1, n_epochs, avg_train_loss, val_loss, val_metrics['R2']))
            else:
                print("Epoch {}/{}, Train Loss: {:.6e}".format(epoch+1, n_epochs, avg_train_loss))
    
    return {
        'train_losses': train_losses,
        'train_mse_losses': train_mse_losses,
        'train_physics_losses': train_physics_losses,
        'val_losses': val_losses,
        'val_metrics_history': val_metrics_history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }


# ==================== 仿真函数 ====================
def simulate_microtubule(model, init_coords, n_steps=1000):
    """
    使用训练好的PINN模型进行仿真
    """
    model.eval()
    results = []
    
    # 保存初始状态
    results.append(init_coords.cpu().numpy())
    current_coords = init_coords.clone().unsqueeze(0).to(device)
    
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            # 模型预测下一时刻
            q_next_pred = model(current_coords)
            
            # 添加随机噪声（Langevin方程的随机项）
            noise_scale = torch.sqrt(2 * k_B * T * dt / xi)
            noise_scale = noise_scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            noise = torch.randn_like(q_next_pred) * noise_scale
            q_next = q_next_pred + noise
            
            # 固定x坐标到初始位置（与C++代码保持一致）
            q_next[:, :, :, 0] = init_coords[:, :, 0].unsqueeze(0).to(device)
            
            # 更新当前坐标
            current_coords = q_next
            results.append(q_next[0].cpu().numpy())
    
    return np.array(results)


# ==================== 保存结果 ====================
def save_results(results, filename="results_pinn_v2.csv"):
    """保存仿真结果到CSV文件"""
    n_steps, n_filaments, n_subunits, n_dims = results.shape
    
    with open(filename, 'w') as f:
        # 写入表头
        f.write("step,layer,")
        for i in range(n_filaments):
            for k in range(n_dims):
                f.write('Q{}{},'.format(i+1, k+1))
        f.write('\n')
        
        # 写入数据
        for step in range(n_steps):
            for layer in range(n_subunits):
                f.write("{},{},".format(step, layer))
                for i in range(n_filaments):
                    for k in range(n_dims):
                        if k < 3:
                            f.write("{},".format(results[step, i, layer, k] * 1e9))
                        else:
                            f.write("{},".format(results[step, i, layer, k]))
                f.write('\n')
    
    # 文件已保存


# ==================== 主函数 ====================
if __name__ == "__main__":
    # 参数设置
    n_filaments = 13
    n_subunits = 4
    
    # 1. 初始化
    init_coords = initialize_microtubule(n_filaments, n_subunits)
    
    # 2. 创建模型
    model = MicrotubuleDynamicsModel(
        n_filaments=n_filaments, n_subunits=n_subunits,
        hidden_size=128, num_layers=3
    ).to(device)
    print("模型: {}节点, {}边, {}参数".format(
        model.num_nodes, model.edge_index.shape[1], 
        sum(p.numel() for p in model.parameters())))
    
    # 3. 加载数据
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    cpp_data_path = os.path.join(project_root, 'MT_tubulin-master', 'results_5k.csv')
    if not os.path.exists(cpp_data_path):
        cpp_data_path = os.path.join('..', '..', 'MT_tubulin-master', 'results_5k.csv')
    
    all_coords = generate_training_data(init_coords, n_steps=None, use_cpp_data=True, cpp_data_path=cpp_data_path)
    all_coords = all_coords.to(device)
    
    # 4. 划分数据集
    train_indices, val_indices, test_indices = split_dataset(
        all_coords, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print("数据集: 训练{} / 验证{} / 测试{}".format(len(train_indices), len(val_indices), len(test_indices)))
    
    # 5. 训练
    print("\n开始训练...")
    training_history = train_pinn(
        model, init_coords, train_indices, val_indices,
        n_epochs=2000, batch_size=32, lr=1e-3,
        lambda_physics=0.1, save_best=True
    )
    
    # 6. 测试集评估
    best_model_path = "microtubule_pinn_v2_best_model.pt"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\n最佳模型: Epoch {}, Val MSE: {:.6e}".format(
        checkpoint['epoch'] + 1, checkpoint['val_loss']))
    
    physics_constraint = LangevinPhysicsConstraint(n_filaments, n_subunits)
    test_metrics, _, _ = evaluate(
        model, test_indices, all_coords, batch_size=32, physics_constraint=physics_constraint
    )
    
    EvaluationMetrics(n_filaments, n_subunits).print_metrics(test_metrics, "测试集")
    
    # 7. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['val_loss']
    }, "microtubule_pinn_v2_final_model.pt")
    
    # 8. 可选仿真
    run_simulation = input("\n是否运行仿真？(y/n): ").lower().strip()
    if run_simulation == 'y':
        n_steps = int(input("请输入时间步数: "))
        results = simulate_microtubule(model, init_coords, n_steps=n_steps)
        save_results(results, "results_pinn_v2.csv")
    
    # 9. 可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(training_history['train_losses'], label='Train Total Loss', alpha=0.7)
    plt.plot(training_history['train_mse_losses'], label='Train MSE Loss', alpha=0.7)
    plt.plot(training_history['train_physics_losses'], label='Train Physics Loss', alpha=0.7)
    if len(training_history['val_losses']) > 0:
        plt.plot(training_history['val_losses'], label='Val MSE Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    if len(training_history['val_metrics_history']) > 0:
        val_r2 = [m['R2'] for m in training_history['val_metrics_history']]
        plt.plot(val_r2, label='Validation R²', color='green', linewidth=2)
        plt.axvline(x=training_history['best_epoch'], color='red', linestyle='--', 
                   label='Best Model (Epoch {})'.format(training_history['best_epoch'] + 1))
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Validation R² Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    if len(training_history['val_metrics_history']) > 0:
        val_physics = [m.get('Physics_Constraint_Error', 0) for m in training_history['val_metrics_history']]
        if max(val_physics) > 0:
            plt.plot(val_physics, label='Physics Constraint Error', color='orange', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Physics Error')
            plt.title('Physics Constraint Error')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_v2_training_curves.png', dpi=150)
    print("\n完成！")