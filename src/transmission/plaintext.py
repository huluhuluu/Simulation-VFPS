"""
明文传输（无加密）
"""

import torch
from typing import Any
from .base import BaseTransmission, TransmissionConfig


class PlaintextTransmission(BaseTransmission):
    """明文传输 - 无加密，直接传输"""
    
    def __init__(self, config: TransmissionConfig = None):
        super().__init__(config)
        self.method = 'plaintext'
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Any:
        """明文传输不需要加密，直接返回"""
        return tensor.clone()
    
    def decrypt_tensor(self, encrypted_data: Any) -> torch.Tensor:
        """明文传输不需要解密，直接返回"""
        return encrypted_data
