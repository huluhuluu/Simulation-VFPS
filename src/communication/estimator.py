"""
通信时间估算器
"""

import sys
import io
import time
import pickle
import torch
from typing import Dict, Tuple


class CommunicationEstimator:
    """通信时间估算器
    
    使用字典缓存不同张量大小的加密结果：
        {tensor_numel: (encrypt_time, ciphertext_bytes)}
    
    第一次遇到新大小时进行性能测量，后续请求返回缓存值。
    """
    
    def __init__(self, bandwidth_mbps: float = 300, encryption: str = 'plaintext'):
        """初始化估算器
        
        Args:
            bandwidth_mbps: 带宽（Mbps）
            encryption: 加密方式 ('plaintext', 'paillier', 'tenseal')
        """
        self.bandwidth_bps = bandwidth_mbps * 1e6
        self.encryption = encryption
        
        # Cache: {tensor_numel: (encrypt_time, ciphertext_bytes)}
        self._profile_cache: Dict[int, Tuple[float, int]] = {}
        
        # Accumulated data volume
        self.total_bytes = 0
    
    def _profile_encrypt(self, numel: int) -> Tuple[float, int]:
        """测量加密基线，返回 (encrypt_time, ciphertext_bytes)
        
        Args:
            numel: 张量元素数量
            
        Returns:
            (加密时间, 密文字节数)
        """
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
            # 抑制警告输出
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            try:
                if self.encryption == 'paillier':
                    from src.transmission.paillier.paillier import PaillierTransmission
                    encryptor = PaillierTransmission()
                    
                    t0 = time.time()
                    encrypted = encryptor.encrypt_tensor(sample_tensor)
                    encrypt_time = time.time() - t0
                    
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
        
        Args:
            tensor: 要传输的张量
            
        Returns:
            encryption_time + expanded_communication_time
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
        
        Args:
            tensor: 要传输的张量
            
        Returns:
            plaintext_communication_time
        """
        numel = tensor.numel()
        plaintext_bytes = numel * tensor.element_size()
        
        # 传输时间
        transfer_time = plaintext_bytes * 8 / self.bandwidth_bps
        
        # 累计数据量
        self.total_bytes += plaintext_bytes
        
        return transfer_time
    
    @property
    def total_data_mb(self) -> float:
        """累计传输数据量（MB）"""
        return self.total_bytes / (1024 * 1024)
