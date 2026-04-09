"""
数据分发器 - 垂直联邦学习
"""

import random
import torch
from typing import List, Dict, Tuple, Any


class DataDistributor:
    """数据分发器 - 用于垂直联邦学习
    
    基于 vflweight 设计：按列切分
    - 每个客户端接收 28x(width) 图像
    - 高度固定为 28，宽度可变
    """
    
    def __init__(self, n_clients: int, data_loader, device, test_loader=None):
        """初始化数据分发器
        
        Args:
            n_clients: 客户端数量
            data_loader: 训练数据加载器
            device: 计算设备
            test_loader: 测试数据加载器（可选）
        """
        self.n_clients = n_clients
        self.device = device
        
        # 垂直划分训练数据（按列）
        self.data_pointer: List[Dict[str, torch.Tensor]] = []
        self.labels: List[torch.Tensor] = []
        
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
        self.subdata: List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]] = []
    
    def _create_test_set(self, test_loader) -> List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """使用真正的测试数据创建测试集
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            测试集列表
        """
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
        """生成训练子数据
        
        Args:
            prob: 采样概率
        """
        self.subdata = []
        for idx, (data_ptr, label) in enumerate(zip(self.data_pointer, self.labels)):
            if random.random() <= prob:
                self.subdata.append((idx, data_ptr, label))
    
    def generate_estimate_subdata(self, n_samples: int = 50) -> List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]]:
        """生成用于互信息估计的子数据
        
        Args:
            n_samples: 采样数量
            
        Returns:
            子数据列表
        """
        n_samples = min(n_samples, len(self.data_pointer))
        indices = random.sample(range(len(self.data_pointer)), n_samples)
        return [(idx, self.data_pointer[idx], self.labels[idx]) for idx in indices]
    
    @property
    def n_batches(self) -> int:
        """批次数量"""
        return len(self.data_pointer)
