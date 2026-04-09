"""
TenSEAL/CKKS 同态加密传输实现

CKKS (Cheon-Kim-Kim-Song) 是一种近似同态加密方案，支持：
- 加法同态：E(a) + E(b) = E(a + b)
- 乘法同态：E(a) * E(b) = E(a * b)
- 支持浮点数运算
- 支持向量/矩阵运算
"""

import torch
import numpy as np
from typing import Any, Tuple
import time
import sys
import io

# 尝试导入 tenseal 库
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("Warning: tenseal library not installed. Install with: pip install tenseal")

from ..base import BaseTransmission, TransmissionConfig


def _suppress_tenseal_warnings(func, *args, **kwargs):
    """抑制 TenSEAL 警告的辅助函数"""
    # 保存原始 stderr
    old_stderr = sys.stderr
    # 重定向到 StringIO
    sys.stderr = io.StringIO()
    try:
        result = func(*args, **kwargs)
    finally:
        # 恢复原始 stderr
        sys.stderr = old_stderr
    return result


class TenSEALTransmission(BaseTransmission):
    """TenSEAL/CKKS 同态加密传输"""
    
    def __init__(self, config: TransmissionConfig = None):
        super().__init__(config)
        self.method = 'tenseal'
        
        if not TENSEAL_AVAILABLE:
            raise ImportError("tenseal library is required for CKKS encryption. Install with: pip install tenseal")
        
        # 创建 CKKS 上下文（抑制警告）
        self.context = _suppress_tenseal_warnings(
            ts.context,
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.config.poly_modulus_degree,
            coeff_mod_bit_sizes=self.config.coeff_mod_bit_sizes
        )
        
        # 生成密钥
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        
        # 保存上下文用于解密
        self.serialize_context = self.context.serialize(save_secret_key=True)
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Any:
        """
        加密张量
        
        Args:
            tensor: 要加密的张量
            
        Returns:
            加密后的数据
        """
        # 转换为 numpy
        tensor_np = tensor.detach().cpu().numpy()
        flat_tensor = tensor_np.flatten().tolist()
        
        # 使用 CKKS 加密（抑制警告）
        encrypted_tensor = _suppress_tenseal_warnings(
            ts.ckks_vector, self.context, flat_tensor
        )
        
        # 返回加密数据和元信息
        return {
            'encrypted_data': encrypted_tensor.serialize(),
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
        # 反序列化上下文和加密数据
        context = ts.context_from(self.serialize_context)
        encrypted_vector = ts.ckks_vector_from(context, encrypted_data['encrypted_data'])
        
        # 解密
        decrypted_list = encrypted_vector.decrypt()
        
        # 转换回张量
        shape = encrypted_data['shape']
        decrypted_np = np.array(decrypted_list).reshape(shape)
        return torch.from_numpy(decrypted_np).float()
    
    def encrypt_matrix(self, tensor: torch.Tensor) -> Any:
        """
        加密矩阵（支持 2D 张量）
        
        使用 CKKS 矩阵加密，更高效
        """
        if tensor.dim() != 2:
            raise ValueError("encrypt_matrix only supports 2D tensors")
        
        tensor_np = tensor.detach().cpu().numpy()
        
        # 加密矩阵
        encrypted_matrix = ts.ckks_tensor(self.context, tensor_np)
        
        return {
            'encrypted_data': encrypted_matrix.serialize(),
            'shape': tensor_np.shape,
            'dtype': str(tensor.dtype)
        }
    
    def decrypt_matrix(self, encrypted_data: dict) -> torch.Tensor:
        """解密矩阵"""
        context = ts.context_from(self.serialize_context)
        encrypted_tensor = ts.ckks_tensor_from(context, encrypted_data['encrypted_data'])
        
        decrypted_np = encrypted_tensor.decrypt()
        return torch.from_numpy(decrypted_np).float()
    
    def encrypt_add(self, encrypted_a: bytes, encrypted_b: bytes) -> bytes:
        """
        同态加法：E(a) + E(b) = E(a + b)
        """
        context = ts.context_from(self.serialize_context)
        vec_a = ts.ckks_vector_from(context, encrypted_a)
        vec_b = ts.ckks_vector_from(context, encrypted_b)
        
        result = vec_a + vec_b
        return result.serialize()
    
    def encrypt_multiply(self, encrypted_a: bytes, encrypted_b: bytes) -> bytes:
        """
        同态乘法：E(a) * E(b) = E(a * b)
        """
        context = ts.context_from(self.serialize_context)
        vec_a = ts.ckks_vector_from(context, encrypted_a)
        vec_b = ts.ckks_vector_from(context, encrypted_b)
        
        result = vec_a * vec_b
        return result.serialize()
