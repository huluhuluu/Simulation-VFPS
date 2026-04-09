#!/usr/bin/env python3
"""
Dynamic-VFPS GPU 测试脚本
基于互信息的垂直联邦学习动态参与者选择

运行方式:
    python test_gpu.py                                    # 默认参数
    python test_gpu.py --epochs 50 --clients 10          # 自定义参数
    python test_gpu.py --encryption paillier             # 使用 Paillier 加密
    python test_gpu.py --help                            # 查看所有参数
"""

import sys
sys.path.append('./')

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import random
import time
import argparse
import math
import numpy as np

from src.transmission import get_transmission
from src.models.split_resnet import SplitResNet18


# =============================================================================
# 配置类
# =============================================================================

class Config:
    """训练配置"""
    def __init__(self):
        # 训练参数
        self.epochs = 50
        self.learning_rate = 0.001
        self.batch_size = 256
        self.local_epochs = 1  # 本地迭代次数
        self.subset_update_prob = 0.2
        
        # 客户端参数
        self.n_clients = 10
        self.n_selected = 6
        
        # 互信息估计参数
        self.n_tests = 5
        self.k_nn = 3
        
        # 通信参数
        self.bandwidth_mbps = 300
        self.padding_method = "zeros"
        
        # 加密方法
        self.encryption = "plaintext"
        
        # 模型参数
        self.feature_dim = 256
        self.hidden_dim = 128
        self.num_classes = 10
        
        # 评估参数
        self.eval_every_steps = 10
        self.test_set_size = 50
        self.estimate_samples = 50
    
    @classmethod
    def from_args(cls, args):
        """从命令行参数创建配置"""
        config = cls()
        config.epochs = args.epochs
        config.local_epochs = args.local_epochs
        config.n_clients = args.clients
        config.n_selected = args.selected
        config.n_tests = args.n_tests
        config.k_nn = args.k_nn
        config.encryption = args.encryption
        config.bandwidth_mbps = args.bandwidth
        return config
    
    def __str__(self):
        return (
            f"Epochs: {self.epochs}, Local epochs: {self.local_epochs}, "
            f"Clients: {self.n_clients}/{self.n_selected}, "
            f"Encryption: {self.encryption}, Bandwidth: {self.bandwidth_mbps} Mbps"
        )


# =============================================================================
# 工具函数
# =============================================================================

def digamma(x):
    """Digamma 函数 (Gamma 函数的对数导数)"""
    if x == 0:
        return float('-inf')
    return math.log(x) - 0.5 / x


def get_device():
    """获取计算设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


# =============================================================================
# 通信时间估算器
# =============================================================================

class CommunicationEstimator:
    """通信时间估算器
    
    用 dict 缓存不同张量大小的加密结果：
        {tensor_numel: (encrypt_time, ciphertext_bytes)}
    
    首次遇到某个大小时进行 profiler 测量，后续直接查表返回
    """
    
    def __init__(self, bandwidth_mbps: float = 300, encryption: str = 'plaintext'):
        self.bandwidth_bps = bandwidth_mbps * 1e6
        self.encryption = encryption
        
        # 缓存: {tensor_numel: (encrypt_time, ciphertext_bytes)}
        self._profile_cache = {}
        
        # 累计统计
        self.total_time = 0.0
        self.total_bytes = 0
    
    def _profile_encrypt(self, numel: int) -> tuple:
        """测量加密基准，返回 (encrypt_time, ciphertext_bytes)"""
        if numel in self._profile_cache:
            return self._profile_cache[numel]
        
        if self.encryption == 'plaintext':
            # 明文：无加密时间，密文大小 = 明文大小
            plaintext_bytes = numel * 4  # float32
            self._profile_cache[numel] = (0.0, plaintext_bytes)
            return self._profile_cache[numel]
        
        # 实际测量
        sample_tensor = torch.randn(numel, dtype=torch.float32)
        plaintext_bytes = sample_tensor.element_size() * sample_tensor.numel()
        
        print(f"[Profile] Measuring {self.encryption} for numel={numel}...")
        
        try:
            import sys
            import io
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            try:
                if self.encryption == 'paillier':
                    from src.transmission.paillier.paillier import PaillierTransmission
                    encryptor = PaillierTransmission()
                    
                    t0 = time.time()
                    encrypted = encryptor.encrypt_tensor(sample_tensor)
                    encrypt_time = time.time() - t0
                    
                    import pickle
                    ciphertext_bytes = len(pickle.dumps(encrypted['encrypted_data']))
                    
                elif self.encryption == 'tenseal':
                    from src.transmission.tenseal.tenseal import TenSEALTransmission
                    encryptor = TenSEALTransmission()
                    
                    t0 = time.time()
                    encrypted = encryptor.encrypt_tensor(sample_tensor)
                    encrypt_time = time.time() - t0
                    
                    ciphertext_bytes = len(encrypted['encrypted_data'])
                else:
                    encrypt_time = 0.0
                    ciphertext_bytes = plaintext_bytes
            finally:
                sys.stderr = old_stderr
            
            print(f"[Profile] {numel} elements: {encrypt_time:.3f}s encrypt, "
                  f"{plaintext_bytes/1024:.1f}KB -> {ciphertext_bytes/1024:.1f}KB "
                  f"({ciphertext_bytes/plaintext_bytes:.1f}x)")
            
        except Exception as e:
            print(f"[Profile] Failed: {e}, using fallback values")
            # 回退值
            if self.encryption == 'paillier':
                encrypt_time = plaintext_bytes * 2.6e-6
                ciphertext_bytes = int(plaintext_bytes * 139.5)
            elif self.encryption == 'tenseal':
                encrypt_time = plaintext_bytes * 0.3e-6
                ciphertext_bytes = int(plaintext_bytes * 20.4)
            else:
                encrypt_time = 0.0
                ciphertext_bytes = plaintext_bytes
        
        self._profile_cache[numel] = (encrypt_time, ciphertext_bytes)
        return encrypt_time, ciphertext_bytes
    
    def estimate_encrypted(self, tensor: torch.Tensor) -> float:
        """估算加密传输时间（客户端选择阶段）
        
        Returns:
            加密时间 + 膨胀后的通信时间
        """
        numel = tensor.numel()
        encrypt_time, ciphertext_bytes = self._profile_encrypt(numel)
        
        # 传输时间
        transfer_time = ciphertext_bytes * 8 / self.bandwidth_bps
        
        # 累计数据量
        self.total_bytes += ciphertext_bytes
        
        return encrypt_time + transfer_time
    
    def estimate_plaintext(self, tensor: torch.Tensor) -> float:
        """估算明文传输时间（模型训练阶段）
        
        Returns:
            明文通信时间
        """
        numel = tensor.numel()
        plaintext_bytes = numel * tensor.element_size()
        
        # 传输时间
        transfer_time = plaintext_bytes * 8 / self.bandwidth_bps
        
        # 累计数据量
        self.total_bytes += plaintext_bytes
        
        return transfer_time
    
    def add_time(self, t: float):
        """累加时间"""
        self.total_time += t
    
    @property
    def total_data_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)
    
    def reset(self):
        self.total_time = 0.0
        self.total_bytes = 0


# =============================================================================
# 数据分发器
# =============================================================================

class DataDistributor:
    """垂直联邦学习数据分发器
    
    参考 vflweight 的设计：按列分割
    - 每个客户端收到 28x(width) 的图像
    - 高度固定为 28，宽度可变
    """
    
    def __init__(self, n_clients: int, data_loader, device, test_loader=None):
        self.n_clients = n_clients
        self.device = device
        
        # 垂直划分训练数据（按列）
        self.data_pointer = []
        self.labels = []
        
        for images, labels in data_loader:
            images = images.to(device)
            width = images.shape[-1] // n_clients
            
            curr_data = {}
            for i in range(n_clients):
                start_col = i * width
                end_col = start_col + width
                # 按列分割: (batch, 1, 28, 28) -> (batch, 28, width) -> (batch, 28*width)
                image_part = images[:, :, :, start_col:end_col].squeeze(1)  # (batch, 28, width)
                curr_data[f"client_{i}"] = image_part.reshape(images.size(0), -1)
            
            self.data_pointer.append(curr_data)
            self.labels.append(labels)
        
        # 创建测试集（使用真正的测试数据）
        self.test_set = self._create_test_set(test_loader)
        
        # 训练子数据
        self.subdata = []
    
    def _create_test_set(self, test_loader):
        """使用真正的测试数据创建测试集"""
        if test_loader is None:
            return []
        
        test_set = []
        for images, labels in test_loader:
            images = images.to(self.device)
            width = images.shape[-1] // self.n_clients
            
            curr_data = {}
            for i in range(self.n_clients):
                start_col = i * width
                end_col = start_col + width
                image_part = images[:, :, :, start_col:end_col].squeeze(1)
                curr_data[f"client_{i}"] = image_part.reshape(images.size(0), -1)
            
            test_set.append((curr_data, labels))
        
        return test_set
    
    def generate_subdata(self, prob: float = 0.2):
        """生成训练子数据"""
        self.subdata = []
        for idx, (data_ptr, label) in enumerate(zip(self.data_pointer, self.labels)):
            if random.random() <= prob:
                self.subdata.append((idx, data_ptr, label))
    
    def generate_estimate_subdata(self, n_samples: int = 50):
        """生成用于互信息估计的子数据"""
        n_samples = min(n_samples, len(self.data_pointer))
        indices = random.sample(range(len(self.data_pointer)), n_samples)
        return [(idx, self.data_pointer[idx], self.labels[idx]) for idx in indices]
    
    @property
    def n_batches(self) -> int:
        return len(self.data_pointer)


# =============================================================================
# 带互信息估计的 SplitNN
# =============================================================================

class SplitNN:
    """垂直联邦学习分割神经网络，支持基于互信息的动态客户端选择"""
    
    def __init__(self, models, config, optimizers, transmission, comm_estimator, device):
        self.models = models
        self.config = config
        self.optimizers = optimizers
        self.transmission = transmission
        self.comm_estimator = comm_estimator
        self.device = device
        
        # 客户端选择状态
        self.selected = {f"client_{i}": True for i in range(config.n_clients)}
        
        # MI 估计相关
        self.scores = {}
        
        # Padding 缓存
        self.latest = {}
    
    # -------------------------------------------------------------------------
    # 前向传播
    # -------------------------------------------------------------------------
    
    def predict(self, data_ptr):
        """前向传播（明文传输，用于模型训练阶段）
        
        Returns:
            (pred, comm_time, client_outputs): 
                - pred: 预测结果
                - comm_time: 传输时间（取 max 因为并行）
                - client_outputs: 选中客户端的实际输出张量列表（用于梯度传输估算）
        """
        client_outputs = []
        client_times = []
        
        for i in range(self.config.n_clients):
            client_id = f"client_{i}"
            
            if self.selected[client_id]:
                # 客户端前向
                output = self.models[client_id](data_ptr[client_id])
                
                # 明文传输估算
                t = self.comm_estimator.estimate_plaintext(output)
                client_times.append(t)
                
                # 保存输出（用于后续梯度传输估算）
                client_outputs.append(output)
                
                # 更新 padding 缓存
                self._update_padding_cache(client_id, output)
            else:
                # 使用 padding
                padding = self._get_padding(client_id, data_ptr)
                client_outputs.append(padding)
        
        # 服务器前向
        server_input = torch.cat(client_outputs, dim=1)
        pred = self.models["server"](server_input)
        
        return pred, max(client_times) if client_times else 0.0, client_outputs
    
    def _update_padding_cache(self, client_id, output):
        """更新 padding 缓存"""
        self.latest[client_id] = output.detach().clone()
    
    def _get_padding(self, client_id, data_ptr):
        """获取 padding 张量"""
        batch_size = data_ptr[client_id].size(0)
        
        if self.config.padding_method == "latest" and client_id in self.latest:
            latest = self.latest[client_id]
            # 如果 batch size 匹配，使用 latest
            if latest.size(0) == batch_size:
                return latest
        
        # 默认使用 zeros padding
        return torch.zeros(batch_size, self.config.feature_dim, device=self.device)
    
    # -------------------------------------------------------------------------
    # 训练步骤
    # -------------------------------------------------------------------------
    
    def train_step(self, data_ptr, target, local_epochs: int = 1):
        """单步训练，支持本地多轮迭代
        
        Args:
            data_ptr: 数据指针
            target: 标签
            local_epochs: 本地迭代次数
        
        Returns:
            (loss, train_time, comm_time): 平均损失、训练时间、通信时间
        """
        total_loss = 0.0
        total_comm_time = 0.0
        
        for local_iter in range(local_epochs):
            iter_start = time.time()
            
            # 清零梯度
            for opt in self.optimizers.values():
                opt.zero_grad()
            
            # 前向传播
            pred, fwd_comm_time, client_outputs = self.predict(data_ptr)
            loss = nn.NLLLoss()(pred, target)
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 累计本轮通信时间（前向 + 反向，梯度大小与输出相同）
            total_comm_time += fwd_comm_time * 2
            
            # 更新参数
            for client_id, opt in self.optimizers.items():
                if client_id == "server":
                    continue
                if self.selected.get(client_id, True):
                    opt.step()
            self.optimizers["server"].step()
        
        # 训练时间（计算时间，不含通信）
        train_time = time.time() - iter_start
        
        return total_loss / local_epochs, train_time, total_comm_time    
    def estimate_mi_cuda(self, subdata):
        """CUDA 版本的 KNN 互信息估计
        
        注意：通信时间在 group_testing 中统一计算
        这里只计算 MI 值
        
        Returns:
            mi: 互信息值
        """
        if not subdata:
            return 0.0
        
        # 批量提取特征（使用所有样本）
        features_list = []
        targets = []
        
        with torch.no_grad():
            for _, data_ptr, target in subdata:
                # target 是整个 batch 的标签
                batch_size = target.size(0) if isinstance(target, torch.Tensor) else len(target)
                
                # 对 batch 中的每个样本
                for sample_idx in range(batch_size):
                    # 聚合选中客户端的特征
                    combined_feat = []
                    for i in range(self.config.n_clients):
                        client_id = f"client_{i}"
                        if self.selected[client_id]:
                            # 取单个样本
                            sample_data = data_ptr[client_id][sample_idx:sample_idx+1]
                            feat = self.models[client_id](sample_data)
                            combined_feat.append(feat[0])
                    
                    if combined_feat:
                        features_list.append(torch.cat(combined_feat))
                    
                    # 获取对应的 target
                    t = target[sample_idx].item() if isinstance(target, torch.Tensor) else target[sample_idx]
                    targets.append(t)
        
        n_samples = len(features_list)
        if n_samples == 0:
            return 0.0
        
        # 计算距离矩阵
        features = torch.stack(features_list)  # [n_samples, feature_dim * n_selected]
        dist_matrix = torch.cdist(features, features)
        dist_matrix.fill_diagonal_(float('inf'))  # 排除自身
        
        # 计算 MI
        targets_tensor = torch.tensor(targets, device=self.device)
        mi = 0.0
        
        for idx in range(n_samples):
            target = targets[idx]
            
            # 类别内距离
            class_mask = (targets_tensor == target)
            Nq = class_mask.sum().item()
            
            class_indices = torch.where(class_mask)[0]
            class_dists = dist_matrix[idx, class_indices]
            
            # 第 k 近邻距离
            k = min(self.config.k_nn, len(class_dists))
            if k == 0:
                continue
            
            rho_k = torch.kthvalue(class_dists, k).values.item()
            
            # 统计 mq
            mq = (dist_matrix[idx] < rho_k).sum().item()
            
            # MI 公式
            if mq > 0:
                mi += digamma(n_samples) - digamma(Nq) + digamma(k) - digamma(mq)
        
        return mi / n_samples
    
    # -------------------------------------------------------------------------
    # 组测试
    # -------------------------------------------------------------------------
    
    def group_testing(self, estimate_subdata, n_tests):
        """组测试选择客户端
        
        流程：
        1. 所有客户端并行发送加密数据给服务器（一次通信）
        2. 服务器本地进行 n_tests 次组测试（无额外通信）
        
        Returns:
            (scores, mi_comm_time, mi_comp_time): 客户端分数和通信时间与计算时间
        """
        self.scores = {f"client_{i}": 0.0 for i in range(self.config.n_clients)}
        
        # 1. 计算所有数据的加密通信时间
        # 每个 batch：所有客户端并行发送，取 max
        # 总时间：所有 batch 的通信时间之和
        batch_comm_times = []
        for _, data_ptr, _ in estimate_subdata:
            client_times = []
            for i in range(self.config.n_clients):
                client_id = f"client_{i}"
                t = self.comm_estimator.estimate_encrypted(data_ptr[client_id])
                client_times.append(t)
            # 这个 batch 的通信时间 = max（并行传输）
            batch_comm_times.append(max(client_times) if client_times else 0.0)
        
        # 总通信时间 = 所有 batch 的通信时间之和
        mi_comm_time = sum(batch_comm_times)
        
        mi_start = time.time()
        # 2. 本地进行 n_tests 次组测试（无需额外通信）
        for _ in range(n_tests):
            # 随机生成测试组
            test_group = self._generate_test_group()
            
            # 临时设置选中状态
            original_selected = self.selected.copy()
            self.selected = {f"client_{i}": (f"client_{i}" in test_group) 
                           for i in range(self.config.n_clients)}
            
            # 计算 MI（本地计算，无通信）
            mi = self.estimate_mi_cuda(estimate_subdata)
            
            # 累加分数
            for client_id in test_group:
                self.scores[client_id] += mi
            
            # 恢复选中状态
            self.selected = original_selected
        
        # 平均分数并选择
        self.scores = {k: v / n_tests for k, v in self.scores.items()}
        self._select_top_clients()
        
        return self.scores, mi_comm_time, time.time() - mi_start
    
    def _generate_test_group(self, p=0.5):
        """随机生成测试组"""
        test_group = []
        while len(test_group) < 1:
            for i in range(self.config.n_clients):
                if np.random.rand() < p:
                    test_group.append(f"client_{i}")
        return test_group
    
    def _select_top_clients(self):
        """选择分数最高的客户端"""
        sorted_clients = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        
        self.selected = {f"client_{i}": False for i in range(self.config.n_clients)}
        for client_id, _ in sorted_clients[:self.config.n_selected]:
            self.selected[client_id] = True


# =============================================================================
# 评估函数
# =============================================================================

def evaluate(splitnn: SplitNN, test_set, device):
    """评估模型准确率"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data_ptr, label in test_set:
            label = label.to(device)
            pred, _, _ = splitnn.predict(data_ptr)  # 忽略通信时间和输出
            
            pred_labels = pred.argmax(dim=1).cpu().numpy()
            true_labels = label.cpu().numpy()
            
            correct += (pred_labels == true_labels).sum()
            total += len(true_labels)
    
    return correct / total if total > 0 else 0.0


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Dynamic-VFPS GPU Test')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--local-epochs', type=int, default=1, help='Local iterations per batch')
    
    # 客户端参数
    parser.add_argument('--clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--selected', type=int, default=6, help='Number of selected clients')
    
    # MI 估计参数
    parser.add_argument('--n-tests', type=int, default=5, help='Number of group tests')
    parser.add_argument('--k-nn', type=int, default=3, help='KNN k value')
    parser.add_argument('--mi-mode', type=str, default='dynamic',
                       choices=['dynamic', 'static'],
                       help='MI mode: dynamic (every step) or static (once at start)')
    parser.add_argument('--mi-ratio', type=float, default=1/9,
                       help='Ratio of training data for MI estimation (static mode)')
    
    # 通信参数
    parser.add_argument('--bandwidth', type=int, default=300, help='Bandwidth in Mbps')
    
    # 加密参数
    parser.add_argument('--encryption', type=str, default='plaintext',
                       choices=['plaintext', 'paillier', 'tenseal'],
                       help='Encryption method')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    config = Config.from_args(args)
    
    # 设备
    device = get_device()
    if torch.cuda.is_available():
        print(f"[INFO] Device: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print(f"[INFO] Device: CPU")
    
    # 打印配置
    print(f"\n{'='*60}")
    print(f"Configuration: {config}")
    print(f"{'='*60}\n")
    
    # 创建通信估算器和加密传输
    comm_estimator = CommunicationEstimator(
        bandwidth_mbps=config.bandwidth_mbps,
        encryption=config.encryption
    )
    transmission = get_transmission(config.encryption)
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = datasets.FashionMNIST(
        root='./datasets/fashion_mnist',
        download=True,
        train=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    
    # 加载真正的测试集
    testset = datasets.FashionMNIST(
        root='./datasets/fashion_mnist',
        download=True,
        train=False,  # 使用测试数据
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Dataset: Fashion-MNIST")
    print(f"  Train: {len(trainset)} samples, {len(trainloader)} batches")
    print(f"  Test:  {len(testset)} samples, {len(testloader)} batches")
    
    # 数据分发
    distributor = DataDistributor(config.n_clients, trainloader, device, testloader)
    print(f"Data distributed: {distributor.n_batches} batches, {config.n_clients} clients")
    
    # 创建模型
    torch.manual_seed(0)
    input_width = 28 // config.n_clients  # 每个客户端的图像宽度
    
    ClientModel, ServerModel = SplitResNet18.create_multi_client_models(
        n_clients=config.n_clients,
        input_width=input_width,
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes
    )
    
    models = {f"client_{i}": ClientModel().to(device) for i in range(config.n_clients)}
    models["server"] = ServerModel().to(device)
    
    client_params = sum(p.numel() for p in models["client_0"].parameters())
    server_params = sum(p.numel() for p in models["server"].parameters())
    print(f"Model: Client {client_params:,} params, Server {server_params:,} params")
    
    # 创建优化器
    optimizers = {f"client_{i}": optim.SGD(models[f"client_{i}"].parameters(), 
                                           lr=config.learning_rate, momentum=0.9)
                  for i in range(config.n_clients)}
    optimizers["server"] = optim.SGD(models["server"].parameters(), 
                                     lr=config.learning_rate, momentum=0.9)
    
    # 创建 SplitNN
    splitnn = SplitNN(models, config, optimizers, transmission, comm_estimator, device)
    
    # -------------------------------------------------------------------------
    # 训练循环
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[Training Started]")
    print(f"{'='*60}")
    print(f"MI Mode: {args.mi_mode}")
    
    total_train_time = 0.0
    total_mi_compute_time = 0.0  # MI 计算时间
    total_mi_comm_time = 0.0     # MI 通信时间
    total_comm_time = 0.0        # 模型训练通信时间
    global_step = 0
    
    # 打印数据维度信息（只在第一个 step 打印一次）
    dim_info_printed = False
    
    # =========================================================================
    # Static 模式：训练前一次性选择客户端
    # =========================================================================
    if args.mi_mode == 'static':
        print(f"\n[Static MI Mode]")
        print(f"  MI data ratio: {args.mi_ratio}")
        
        # 1. 从训练集取 mi_ratio 比例的数据用于 MI 计算
        n_mi_batches = int(len(distributor.data_pointer) * args.mi_ratio)
        mi_indices = random.sample(range(len(distributor.data_pointer)), n_mi_batches)
        mi_data = [(idx, distributor.data_pointer[idx], distributor.labels[idx]) 
                   for idx in mi_indices]
        print(f"  MI batches: {n_mi_batches}")
        
        # 2. 用所有 MI 数据进行组测试选择客户端
        scores, mi_comm_time, mi_compute_time = splitnn.group_testing(mi_data, config.n_tests)
        
        selected = [k for k, v in splitnn.selected.items() if v]
        print(f"  Selected clients: {selected}")
        print(f"  MI total time: {mi_compute_time + mi_comm_time:.2f}s")
        print(f"  MI compute time: {mi_compute_time:.2f}s")
        print(f"  MI comm time: {mi_comm_time:.4f}s ({n_mi_batches} batches)")
        print(f"\n  Clients fixed for all training steps")
        
        total_mi_compute_time = mi_compute_time
        total_mi_comm_time = mi_comm_time
    
    # =========================================================================
    # 训练循环
    # =========================================================================
    for epoch in range(config.epochs):
        # 生成本轮训练数据
        distributor.generate_subdata(config.subset_update_prob)
        
        print(f"\n[Epoch {epoch+1}/{config.epochs}]")
        
        epoch_train_time = 0.0
        epoch_comm_time = 0.0
        
        # 训练
        for _, data_ptr, label in distributor.subdata:
            # Dynamic 模式：每个 step 前进行客户端选择
            if args.mi_mode == 'dynamic':
                estimate_data = distributor.generate_estimate_subdata(config.estimate_samples)
                scores, mi_comm_time, mi_compute_time = splitnn.group_testing(estimate_data, config.n_tests)
                
                total_mi_compute_time += mi_compute_time
                total_mi_comm_time += mi_comm_time
                
                selected = [k for k, v in splitnn.selected.items() if v]
            
            label = label.to(device)
            
            # 打印维度信息（只打印一次）
            if not dim_info_printed:
                print(f"\n{'='*60}")
                print("[Data Dimensions]")
                print(f"{'='*60}")
                for i in range(config.n_clients):
                    client_id = f"client_{i}"
                    input_shape = data_ptr[client_id].shape
                    input_size = input_shape[0] * input_shape[1]
                    print(f"  {client_id} input: {tuple(input_shape)} = {input_size} elements")
                print(f"\n  [After client forward (activation to transmit)]")
                for i in range(config.n_clients):
                    client_id = f"client_{i}"
                    with torch.no_grad():
                        output = splitnn.models[client_id](data_ptr[client_id])
                        output_shape = output.shape
                        output_size = output_shape[0] * output_shape[1]
                        print(f"  {client_id} activation: {tuple(output_shape)} = {output_size} elements")
                print(f"\n  [Server receives from {len(selected)} selected clients]")
                total_activation_size = 0
                for client_id in selected:
                    with torch.no_grad():
                        output = splitnn.models[client_id](data_ptr[client_id])
                        total_activation_size += output.numel()
                print(f"  Total activation size: {total_activation_size} elements = {total_activation_size * 4 / 1024:.2f} KB")
                print(f"{'='*60}\n")
                dim_info_printed = True
            
            loss, train_time, comm_time = splitnn.train_step(data_ptr, label, config.local_epochs)
            
            global_step += 1
            epoch_train_time += train_time
            epoch_comm_time += comm_time
            total_train_time += train_time
            total_comm_time += comm_time
            
            # 定期评估
            if global_step % config.eval_every_steps == 0:
                acc = evaluate(splitnn, distributor.test_set[:10], device)
                data_mb = comm_estimator.total_data_mb
                
                # 整体累计时间
                overall_total_time = (total_train_time + total_comm_time + 
                                     total_mi_compute_time + total_mi_comm_time)
                
                print(f"  Step {global_step:4d} | Selected: {selected} | Loss: {loss:.4f} | Acc: {acc*100:5.2f}%")
                
                # Dynamic 模式显示每步的 MI 时间
                if args.mi_mode == 'dynamic':
                    print(f"         Step: {train_time + comm_time + mi_compute_time + mi_comm_time:.3f}s")
                    print(f"           - Train: {train_time:.3f}s")
                    print(f"           - Comm: {comm_time:.4f}s")
                    print(f"           - MI Compute: {mi_compute_time:.3f}s")
                    print(f"           - MI Comm: {mi_comm_time:.4f}s")
                else:
                    print(f"         Step: {train_time + comm_time:.3f}s")
                    print(f"           - Train: {train_time:.3f}s")
                    print(f"           - Comm: {comm_time:.4f}s")
                
                print(f"         Total: {overall_total_time:.2f}s")
                print(f"           - Train: {total_train_time:.2f}s")
                print(f"           - Comm: {total_comm_time:.2f}s")
                print(f"           - MI Compute: {total_mi_compute_time:.2f}s")
                print(f"           - MI Comm: {total_mi_comm_time:.2f}s")
    
    # -------------------------------------------------------------------------
    # 最终评估
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[Final Evaluation]")
    print(f"{'='*60}")
    
    final_acc = evaluate(splitnn, distributor.test_set, device)
    total_time = (total_train_time + total_comm_time + 
                 total_mi_compute_time + total_mi_comm_time)
    
    print(f"Accuracy: {final_acc*100:.2f}%")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  - Training Time: {total_train_time:.2f}s")
    print(f"  - Communication Time: {total_comm_time:.2f}s")
    print(f"  - MI Compute Time: {total_mi_compute_time:.2f}s")
    print(f"  - MI Comm Time: {total_mi_comm_time:.2f}s")
    print(f"Data Transferred: {comm_estimator.total_data_mb:.2f} MB")
    print(f"Encryption: {config.encryption}")
    print(f"MI Mode: {args.mi_mode}")
    print(f"Local Epochs: {config.local_epochs}")
    print(f"Total Steps: {global_step}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
