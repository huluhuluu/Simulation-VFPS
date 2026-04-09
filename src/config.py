"""
训练配置模块
"""

import argparse


class Config:
    """训练配置"""
    
    def __init__(self):
        # Training parameters
        self.epochs = 50
        self.learning_rate = 0.001
        self.batch_size = 256
        self.local_epochs = 1  # Local iterations per batch
        self.subset_update_prob = 0.2
        
        # Client parameters
        self.n_clients = 10
        self.n_selected = 6
        
        # Mutual information estimation parameters
        self.n_tests = 5
        self.k_nn = 3
        
        # Communication parameters
        self.bandwidth_mbps = 300
        self.padding_method = "zeros"
        
        # Encryption method
        self.encryption = "plaintext"
        
        # Model parameters
        self.feature_dim = 256
        self.hidden_dim = 128
        self.num_classes = 10
        
        # Evaluation parameters
        self.eval_every_steps = 10
        self.estimate_samples = 50
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """从命令行参数创建配置
        
        Args:
            args: 命令行参数
            
        Returns:
            Config 实例
        """
        config = cls()
        config.epochs = args.epochs
        config.learning_rate = args.lr
        config.batch_size = args.batch_size
        config.local_epochs = args.local_epochs
        config.n_clients = args.clients
        config.n_selected = args.selected
        config.n_tests = args.n_tests
        config.k_nn = args.k_nn
        config.encryption = args.encryption
        config.bandwidth_mbps = args.bandwidth
        return config
    
    def __str__(self) -> str:
        return (
            f"Epochs: {self.epochs}, Local epochs: {self.local_epochs}, "
            f"Clients: {self.n_clients}/{self.n_selected}, "
            f"Encryption: {self.encryption}, Bandwidth: {self.bandwidth_mbps} Mbps"
        )
