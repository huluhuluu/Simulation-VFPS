"""
Paillier 同态加密传输实现

Paillier 是一种加法同态加密方案，支持：
- 加法同态：E(a) + E(b) = E(a + b)
- 标量乘法：E(a) * c = E(a * c)
"""

import torch
import numpy as np
from typing import Any, List, Tuple
import time

# 尝试导入 phe 库
try:
    from phe import paillier
    PHE_AVAILABLE = True
except ImportError:
    PHE_AVAILABLE = False
    print("Warning: phe library not installed. Install with: pip install phe")

from ..base import BaseTransmission, TransmissionConfig


class PaillierTransmission(BaseTransmission):
    """Paillier 同态加密传输"""
    
    def __init__(self, config: TransmissionConfig = None):
        super().__init__(config)
        self.method = 'paillier'
        
        if not PHE_AVAILABLE:
            raise ImportError("phe library is required for Paillier encryption. Install with: pip install phe")
        
        # 生成密钥对
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=self.config.key_size
        )
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> List[Any]:
        """
        加密张量
        
        Args:
            tensor: 要加密的张量
            
        Returns:
            加密后的数据列表
        """
        # 转换为 numpy
        tensor_np = tensor.detach().cpu().numpy()
        flat_tensor = tensor_np.flatten()
        
        # 加密每个元素
        encrypted_list = []
        for value in flat_tensor:
            encrypted_value = self.public_key.encrypt(float(value))
            encrypted_list.append(encrypted_value)
        
        # 返回加密数据和原始形状
        return {
            'encrypted_data': encrypted_list,
            'shape': tensor_np.shape,
            'dtype': str(tensor.dtype)
        }
    
    def decrypt_tensor(self, encrypted_data: dict) -> torch.Tensor:
        """
        解密张量
        
        Args:
            encrypted_data: 加密的数据字典
            
        Returns:
            解密后的张量
        """
        encrypted_list = encrypted_data['encrypted_data']
        shape = encrypted_data['shape']
        
        # 解密每个元素
        decrypted_list = []
        for encrypted_value in encrypted_list:
            decrypted_value = self.private_key.decrypt(encrypted_value)
            decrypted_list.append(decrypted_value)
        
        # 转换回张量
        decrypted_np = np.array(decrypted_list).reshape(shape)
        return torch.from_numpy(decrypted_np).float()
    
    def encrypt_add(self, encrypted_a: List[Any], encrypted_b: List[Any]) -> List[Any]:
        """
        同态加法：E(a) + E(b) = E(a + b)
        """
        if len(encrypted_a) != len(encrypted_b):
            raise ValueError("Encrypted tensors must have the same length")
        
        result = []
        for ea, eb in zip(encrypted_a, encrypted_b):
            result.append(ea + eb)
        
        return result
    
    def encrypt_scalar_multiply(self, encrypted: List[Any], scalar: float) -> List[Any]:
        """
        同态标量乘法：E(a) * c = E(a * c)
        """
        result = []
        for e in encrypted:
            result.append(e * scalar)
        
        return result
