"""
加密传输基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple
import torch
import time


@dataclass
class TransmissionConfig:
    """传输配置"""
    method: str = 'plaintext'  # 'plaintext', 'paillier', 'tenseal'
    key_size: int = 2048       # 密钥大小 (Paillier)
    poly_modulus_degree: int = 8192  # 多项式模数次数 (CKKS)
    coeff_mod_bit_sizes: list = None  # 系数模数位大小 (CKKS)
    
    def __post_init__(self):
        if self.coeff_mod_bit_sizes is None:
            self.coeff_mod_bit_sizes = [60, 40, 40, 60]


class BaseTransmission(ABC):
    """加密传输基类"""
    
    def __init__(self, config: TransmissionConfig = None):
        self.config = config or TransmissionConfig()
        self.encrypt_time = 0.0
        self.decrypt_time = 0.0
        self.transfer_time = 0.0
    
    @abstractmethod
    def encrypt_tensor(self, tensor: torch.Tensor) -> Any:
        """
        加密张量
        
        Args:
            tensor: 要加密的张量
            
        Returns:
            加密后的数据
        """
        pass
    
    @abstractmethod
    def decrypt_tensor(self, encrypted_data: Any) -> torch.Tensor:
        """
        解密张量
        
        Args:
            encrypted_data: 加密的数据
            
        Returns:
            解密后的张量
        """
        pass
    
    def transmit(self, tensor: torch.Tensor, simulate_delay: float = 0.0) -> Tuple[torch.Tensor, dict]:
        """
        传输张量（加密 -> 传输 -> 解密）
        
        Args:
            tensor: 要传输的张量
            simulate_delay: 模拟网络延迟（秒）
            
        Returns:
            (解密后的张量, 时间统计字典)
        """
        timings = {}
        
        # 加密
        t0 = time.time()
        encrypted = self.encrypt_tensor(tensor)
        timings['encrypt_time'] = time.time() - t0
        self.encrypt_time += timings['encrypt_time']
        
        # 模拟传输延迟
        if simulate_delay > 0:
            time.sleep(simulate_delay)
        timings['transfer_time'] = simulate_delay
        self.transfer_time += timings['transfer_time']
        
        # 解密
        t0 = time.time()
        decrypted = self.decrypt_tensor(encrypted)
        timings['decrypt_time'] = time.time() - t0
        self.decrypt_time += timings['decrypt_time']
        
        timings['total_time'] = timings['encrypt_time'] + timings['transfer_time'] + timings['decrypt_time']
        
        return decrypted, timings
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'encrypt_time': self.encrypt_time,
            'decrypt_time': self.decrypt_time,
            'transfer_time': self.transfer_time,
            'total_time': self.encrypt_time + self.decrypt_time + self.transfer_time
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.encrypt_time = 0.0
        self.decrypt_time = 0.0
        self.transfer_time = 0.0
