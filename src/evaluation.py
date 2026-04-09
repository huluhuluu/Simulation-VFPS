"""
模型评估模块
"""

import torch
from typing import List, Tuple, Dict

from src.splitnn import SplitNN


def evaluate(splitnn: SplitNN, test_set: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]], 
             device: torch.device) -> float:
    """评估模型准确率
    
    Args:
        splitnn: 分割神经网络
        test_set: 测试集
        device: 计算设备
        
    Returns:
        准确率
    """
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
