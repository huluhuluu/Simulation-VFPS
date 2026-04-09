"""
工具函数模块
"""

import argparse
import math
import torch


def digamma(x: float) -> float:
    """Digamma 函数 (Gamma 函数的对数导数)
    
    Args:
        x: 输入值
        
    Returns:
        Digamma 函数值
    """
    if x == 0:
        return float('-inf')
    return math.log(x) - 0.5 / x


def get_device() -> torch.device:
    """获取计算设备
    
    Returns:
        torch.device: CUDA 设备（如果可用）否则返回 CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def parse_args() -> argparse.Namespace:
    """解析命令行参数
    
    Returns:
        解析后的参数命名空间
    """
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
