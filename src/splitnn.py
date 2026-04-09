"""
分割神经网络 - 垂直联邦学习
"""

import time
import torch
import numpy as np
from torch import nn
from typing import Dict, List, Tuple, Any

from src.config import Config
from src.communication.estimator import CommunicationEstimator
from src.utils.helpers import digamma


class SplitNN:
    """分割神经网络 - 用于垂直联邦学习，基于 MI 的动态客户端选择"""
    
    def __init__(self, models, config: Config, optimizers, 
                 comm_estimator: CommunicationEstimator, device):
        """初始化分割神经网络
        
        Args:
            models: 模型字典
            config: 训练配置
            optimizers: 优化器字典
            comm_estimator: 通信估算器
            device: 计算设备
        """
        self.models = models
        self.config = config
        self.optimizers = optimizers
        self.comm_estimator = comm_estimator
        self.device = device
        
        # 客户端选择状态
        self.selected = {f"client_{i}": True for i in range(config.n_clients)}
        
        # MI 估计相关
        self.scores: Dict[str, float] = {}
        
        # Padding cache
        self.latest: Dict[str, torch.Tensor] = {}
    
    # -------------------------------------------------------------------------
    # Forward Propagation
    # -------------------------------------------------------------------------
    
    def predict(self, data_ptr: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float, List[torch.Tensor]]:
        """前向传播（明文传输，用于模型训练阶段）
        
        Args:
            data_ptr: 数据指针
            
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
                
                # Update padding cache
                self._update_padding_cache(client_id, output)
            else:
                # Use padding
                padding = self._get_padding(client_id, data_ptr)
                client_outputs.append(padding)
        
        # 服务器前向
        server_input = torch.cat(client_outputs, dim=1)
        pred = self.models["server"](server_input)
        
        return pred, max(client_times) if client_times else 0.0, client_outputs
    
    def _update_padding_cache(self, client_id: str, output: torch.Tensor):
        """更新 padding 缓存"""
        self.latest[client_id] = output.detach().clone()
    
    def _get_padding(self, client_id: str, data_ptr: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取 padding 张量"""
        batch_size = data_ptr[client_id].size(0)
        
        if self.config.padding_method == "latest" and client_id in self.latest:
            latest = self.latest[client_id]
            # 如果 batch size 匹配，使用 latest
            if latest.size(0) == batch_size:
                return latest
        
        # Default: zeros padding
        return torch.zeros(batch_size, self.config.feature_dim, device=self.device)
    
    # -------------------------------------------------------------------------
    # Training Step
    # -------------------------------------------------------------------------
    
    def train_step(self, data_ptr: Dict[str, torch.Tensor], target: torch.Tensor, 
                   local_epochs: int = 1) -> Tuple[float, float, float]:
        """单个训练步骤（包含本地迭代）
        
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
            
            # Zero gradients
            for opt in self.optimizers.values():
                opt.zero_grad()
            
            # Forward propagation
            pred, fwd_comm_time, client_outputs = self.predict(data_ptr)
            loss = nn.NLLLoss()(pred, target)
            total_loss += loss.item()
            
            # Backward propagation
            loss.backward()
            
            # Accumulate communication time for this round (forward + backward, gradient size same as output)
            total_comm_time += fwd_comm_time * 2
            
            # Update parameters
            for client_id, opt in self.optimizers.items():
                if client_id == "server":
                    continue
                if self.selected.get(client_id, True):
                    opt.step()
            self.optimizers["server"].step()
        
        # 训练时间（计算时间，不含通信）
        train_time = time.time() - iter_start
        
        return total_loss / local_epochs, train_time, total_comm_time
    
    # -------------------------------------------------------------------------
    # Mutual Information Estimation
    # -------------------------------------------------------------------------
    
    def estimate_mi_cuda(self, subdata: List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]]) -> float:
        """CUDA 版本的 KNN 互信息估计
        
        注意：通信时间在 group_testing 中计算
        此方法仅计算 MI 值
        
        Args:
            subdata: 子数据列表
            
        Returns:
            互信息值
        """
        if not subdata:
            return 0.0
        
        # Batch feature extraction (using all samples)
        features_list = []
        targets = []
        
        with torch.no_grad():
            for _, data_ptr, target in subdata:
                # target is the label for entire batch
                batch_size = target.size(0) if isinstance(target, torch.Tensor) else len(target)
                
                # For each sample in the batch
                for sample_idx in range(batch_size):
                    # Aggregate features from selected clients
                    combined_feat = []
                    for i in range(self.config.n_clients):
                        client_id = f"client_{i}"
                        if self.selected[client_id]:
                            # Take single sample
                            sample_data = data_ptr[client_id][sample_idx:sample_idx+1]
                            feat = self.models[client_id](sample_data)
                            combined_feat.append(feat[0])
                    
                    if combined_feat:
                        features_list.append(torch.cat(combined_feat))
                    
                    # Get corresponding target
                    t = target[sample_idx].item() if isinstance(target, torch.Tensor) else target[sample_idx]
                    targets.append(t)
        
        n_samples = len(features_list)
        if n_samples == 0:
            return 0.0
        
        # Compute distance matrix
        features = torch.stack(features_list)  # [n_samples, feature_dim * n_selected]
        dist_matrix = torch.cdist(features, features)
        dist_matrix.fill_diagonal_(float('inf'))  # Exclude self
        
        # Compute MI
        targets_tensor = torch.tensor(targets, device=self.device)
        mi = 0.0
        
        for idx in range(n_samples):
            target = targets[idx]
            
            # In-class distance
            class_mask = (targets_tensor == target)
            Nq = class_mask.sum().item()
            
            class_indices = torch.where(class_mask)[0]
            class_dists = dist_matrix[idx, class_indices]
            
            # k-th nearest neighbor distance
            k = min(self.config.k_nn, len(class_dists))
            if k == 0:
                continue
            
            rho_k = torch.kthvalue(class_dists, k).values.item()
            
            # Count mq
            mq = (dist_matrix[idx] < rho_k).sum().item()
            
            # MI formula
            if mq > 0:
                mi += digamma(n_samples) - digamma(Nq) + digamma(k) - digamma(mq)
        
        return mi / n_samples
    
    # -------------------------------------------------------------------------
    # Group Testing
    # -------------------------------------------------------------------------
    
    def group_testing(self, estimate_subdata: List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]], 
                      n_tests: int) -> Tuple[Dict[str, float], float, float]:
        """组测试进行客户端选择
        
        流程：
        1. 所有客户端并行发送加密数据给服务器（一次通信）
        2. 服务器本地进行 n_tests 次组测试（无额外通信）
        
        Args:
            estimate_subdata: 用于估计的子数据
            n_tests: 组测试次数
            
        Returns:
            (scores, mi_comm_time, mi_comp_time): 客户端分数、通信时间、计算时间
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
    
    def _generate_test_group(self, p: float = 0.5) -> List[str]:
        """随机生成测试组
        
        Args:
            p: 选择概率
            
        Returns:
            测试组客户端 ID 列表
        """
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
