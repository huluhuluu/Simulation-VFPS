"""
加密传输模块

支持多种加密方法：
- plaintext: 明文传输（无加密）
- paillier: Paillier 同态加密
- tenseal: TenSEAL/CKKS 同态加密
"""

from .base import BaseTransmission, TransmissionConfig
from .plaintext import PlaintextTransmission

__all__ = [
    'BaseTransmission',
    'TransmissionConfig', 
    'PlaintextTransmission',
    'get_transmission'
]

def get_transmission(method: str = 'plaintext', **kwargs):
    """
    获取传输方法实例
    
    Args:
        method: 加密方法 ('plaintext', 'paillier', 'tenseal')
        **kwargs: 传递给具体实现的参数
    
    Returns:
        传输方法实例
    """
    if method == 'plaintext':
        return PlaintextTransmission(**kwargs)
    elif method == 'paillier':
        from .paillier import PaillierTransmission
        return PaillierTransmission(**kwargs)
    elif method == 'tenseal':
        from .tenseal import TenSEALTransmission
        return TenSEALTransmission(**kwargs)
    else:
        raise ValueError(f"Unknown transmission method: {method}")
