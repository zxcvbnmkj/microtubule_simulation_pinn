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
    
    print("从C++结果文件加载训练数据: {}".format(cpp_data_path))
    
    if not os.path.exists(cpp_data_path):
        raise FileNotFoundError("找不到C++结果文件: {}".format(cpp_data_path))
    
    # 读取CSV文件
    df = pd.read_csv(cpp_data_path)
    
    # 获取所有唯一的时间步
    unique_steps = sorted(df['step'].unique())
    
    if n_steps is not None:
        unique_steps = unique_steps[:n_steps+1]
    
    print("找到 {} 个时间步".format(len(unique_steps)))
    
    all_coords = []
    n_filaments = 13
    n_subunits = 4
    n_dims = 6
    
    # 遍历每个时间步
    for step in unique_steps:
        # 获取该时间步的所有layer数据
        step_data = df[df['step'] == step].sort_values('layer')
        
        # 检查是否有4个layer
        if len(step_data) != n_subunits:
            print("警告: 时间步 {} 只有 {} 个layer，期望4个".format(step, len(step_data)))
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
    
    # 堆叠所有时间步
    all_coords_tensor = torch.stack(all_coords)
    
    print("成功加载 {} 个时间步的数据".format(len(all_coords_tensor)))
    print("数据形状: {}".format(all_coords_tensor.shape))
    
    return all_coords_tensor


# ==================== 训练函数 ====================
def train_pinn(model, init_coords, n_epochs=1000, batch_size=32, lr=1e-3, lambda_physics=0.1):
    """
    训练PINN模型
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    physics_constraint = LangevinPhysicsConstraint(model.n_filaments, model.n_subunits)
    loss_fn = PINNLoss(physics_constraint, lambda_physics=lambda_physics)
    
    # 从C++结果文件加载训练数据
    print("加载训练数据...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    cpp_data_path = os.path.join(project_root, 'MT_tubulin-master', 'results_5k.csv')
    
    if not os.path.exists(cpp_data_path):
        cpp_data_path = os.path.join('..', '..', 'MT_tubulin-master', 'results_5k.csv')
    
    all_coords = generate_training_data(init_coords, n_steps=None, use_cpp_data=True, cpp_data_path=cpp_data_path)
    all_coords = all_coords.to(device)
    
    losses = []
    mse_losses = []
    physics_losses = []
    
    for epoch in range(n_epochs):
        epoch_total_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_physics_loss = 0.0
        n_batches = 0
        
        # 随机打乱数据
        indices = torch.randperm(len(all_coords) - 1)
        
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start+batch_size]
            
            # 准备批次数据
            q_current_batch = all_coords[batch_indices]
            q_next_true_batch = all_coords[batch_indices + 1]
            
            optimizer.zero_grad()
            
            # 模型预测
            q_next_pred_batch = model(q_current_batch)
            
            # 计算损失
            total_loss, mse_loss_val, physics_loss_val = loss_fn(
                q_next_pred_batch, q_next_true_batch, q_current_batch
            )
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_mse_loss += mse_loss_val.item()
            epoch_physics_loss += physics_loss_val.item()
            n_batches += 1
        
        avg_total_loss = epoch_total_loss / n_batches
        avg_mse_loss = epoch_mse_loss / n_batches
        avg_physics_loss = epoch_physics_loss / n_batches
        
        losses.append(avg_total_loss)
        mse_losses.append(avg_mse_loss)
        physics_losses.append(avg_physics_loss)
        
        if (epoch + 1) % 100 == 0:
            print("Epoch {}/{}, Total Loss: {:.6e}, MSE Loss: {:.6e}, Physics Loss: {:.6e}".format(
                epoch+1, n_epochs, avg_total_loss, avg_mse_loss, avg_physics_loss))
    
    return losses, mse_losses, physics_losses


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
    
    print("结果已保存到 {}".format(filename))


# ==================== 主函数 ====================
if __name__ == "__main__":
    # 参数设置
    n_filaments = 13
    n_subunits = 4
    
    print("=" * 60)
    print("微管动力学PINN仿真系统 V2")
    print("输入: q(t) -> 输出: q(t+1)")
    print("=" * 60)
    
    # 1. 初始化微管结构
    print("\n1. 初始化微管结构...")
    init_coords = initialize_microtubule(n_filaments, n_subunits)
    print("   微管结构: {}条原纤维 × {}个亚基".format(n_filaments, n_subunits))
    
    # 2. 创建图神经网络模型
    print("\n2. 创建图神经网络模型...")
    model = MicrotubuleDynamicsModel(
        n_filaments=n_filaments,
        n_subunits=n_subunits,
        hidden_size=128,
        num_layers=3
    ).to(device)
    print("   图结构: {}个节点, {}条边".format(
        model.num_nodes, 
        model.edge_index.shape[1]
    ))
    print("   模型参数数量: {}".format(sum(p.numel() for p in model.parameters())))
    
    # 3. 训练模型
    print("\n3. 训练PINN模型...")
    print("   损失函数组成:")
    print("   - MSE损失（数据拟合）")
    print("   - 物理损失（Langevin方程约束）")
    print("   - 总损失 = MSE损失 + λ × 物理损失")
    print("   这可能需要一些时间...")
    
    lambda_physics = 0.1  # 物理损失权重
    losses, mse_losses, physics_losses = train_pinn(
        model, init_coords, 
        n_epochs=2000, 
        batch_size=32, 
        lr=1e-3,
        lambda_physics=lambda_physics
    )
    
    # 4. 保存模型
    print("\n4. 保存模型...")
    model_path = "microtubule_pinn_v2_model.pt"
    torch.save(model.state_dict(), model_path)
    print("   模型已保存到 {}".format(model_path))
    
    # 5. 运行仿真
    print("\n5. 运行仿真...")
    n_steps = int(input("请输入时间步数: "))
    results = simulate_microtubule(model, init_coords, n_steps=n_steps)
    
    # 6. 保存结果
    print("\n6. 保存结果...")
    save_results(results, "results_pinn_v2.csv")
    
    # 7. 可视化训练损失
    print("\n7. 可视化训练损失...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Total Loss')
    plt.plot(mse_losses, label='MSE Loss')
    plt.plot(physics_losses, label='Physics Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pinn_v2_training_loss.png', dpi=150)
    print("仿真完成！")