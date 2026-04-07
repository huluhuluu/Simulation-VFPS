"""
ResNet18 分割模型 - 用于垂直联邦学习

提供两种分割方案：
1. 单客户端分割：ResNet18 在客户端，分类头在服务器
2. 多客户端垂直分割：每个客户端处理图像的一部分行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResidualBlock


class ClientNet(nn.Module):
    """
    客户端网络 - 单客户端版本
    
    使用 ResNet18 提取特征
    参考 vflweight 的设计
    """
    
    def __init__(self, feature_dim=1152):
        super(ClientNet, self).__init__()
        from .resnet import ResNet
        self.resnet18 = ResNet(ResidualBlock, num_classes=feature_dim, in_channel=1)
        self.feature_dim = feature_dim

    def forward(self, x):
        # 输入可能是 (batch, 784) 或 (batch, 28, 28) 或 (batch, 1, 28, 28)
        if x.dim() == 2:
            # (batch, 784) -> (batch, 1, 28, 28)
            x = x.view(x.shape[0], 1, 28, -1)
        elif x.dim() == 3:
            # (batch, 28, 28) -> (batch, 1, 28, 28)
            x = x.unsqueeze(1)
        # else: 已经是 (batch, 1, 28, 28)
        
        x = self.resnet18(x)
        return x  # 输出: (batch, feature_dim)


class ServerNet(nn.Module):
    """
    服务器网络 - 单客户端版本
    
    接收客户端特征，完成分类
    """
    
    def __init__(self, input_dim=1152, hidden_dim=64, num_classes=10):
        super(ServerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 输出: (batch, num_classes)


class MultiClientNet(nn.Module):
    """
    客户端网络 - 多客户端垂直分割版本
    
    每个客户端处理图像的一部分行
    """
    
    def __init__(self, input_rows=7, feature_dim=256):
        super(MultiClientNet, self).__init__()
        self.input_rows = input_rows
        self.feature_dim = feature_dim
        
        # 适配小图像片段的 CNN
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, feature_dim)
    
    def forward(self, x):
        # 输入: (batch, 1, 7, 28) 或 (batch, 7, 28) 或 (batch, 196)
        if x.dim() == 2:
            # (batch, 196) -> (batch, 1, 7, 28)
            x = x.view(x.shape[0], 1, self.input_rows, -1)
        elif x.dim() == 3:
            # (batch, 7, 28) -> (batch, 1, 7, 28)
            x = x.unsqueeze(1)
        
        x = self.features(x)  # -> (batch, 128, 7, 28)
        x = self.pool(x)      # -> (batch, 128, 1, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)        # -> (batch, feature_dim)
        return x


class MultiClientServerNet(nn.Module):
    """
    服务器网络 - 多客户端版本
    
    聚合所有客户端特征后分类
    """
    
    def __init__(self, n_clients=4, feature_dim=256, hidden_dim=64, num_classes=10):
        super(MultiClientServerNet, self).__init__()
        total_feature_dim = feature_dim * n_clients
        self.fc1 = nn.Linear(total_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, *client_features):
        # 拼接所有客户端特征
        x = torch.cat(client_features, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SplitResNet18:
    """
    ResNet18 分割模型工厂类
    
    提供创建不同分割方案的静态方法
    """
    
    @staticmethod
    def create_single_client_models(feature_dim=1152, hidden_dim=64, num_classes=10):
        """
        创建单客户端分割模型
        
        Args:
            feature_dim: 特征维度 (默认 1152，参考 vflweight)
            hidden_dim: 服务器隐藏层维度
            num_classes: 分类数
        
        Returns:
            (ClientModelClass, ServerModelClass) 元组 - 返回类而不是实例
        """
        # 返回类，让用户可以实例化多次
        class ClientModelClass(nn.Module):
            def __init__(self):
                super().__init__()
                from .resnet import ResNet, ResidualBlock
                self.resnet18 = ResNet(ResidualBlock, num_classes=feature_dim, in_channel=1)
                self.feature_dim = feature_dim

            def forward(self, x):
                if x.dim() == 2:
                    x = x.view(x.shape[0], 1, 28, -1)
                elif x.dim() == 3:
                    x = x.unsqueeze(1)
                x = self.resnet18(x)
                return x
        
        class ServerModelClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(feature_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                # 添加 LogSoftmax 以配合 NLLLoss
                return F.log_softmax(x, dim=1)
        
        return ClientModelClass, ServerModelClass
    
    @staticmethod
    def create_multi_client_models(n_clients=4, input_rows=7, feature_dim=256, 
                                    hidden_dim=64, num_classes=10):
        """
        创建多客户端垂直分割模型
        
        Args:
            n_clients: 客户端数量
            input_rows: 每个客户端处理的图像行数 (28 / n_clients)
            feature_dim: 每个客户端的特征维度
            hidden_dim: 服务器隐藏层维度
            num_classes: 分类数
        
        Returns:
            (ClientModelClass, ServerModelClass) 元组 - 返回类而不是实例
        """
        # 返回类，让用户可以实例化多次
        class ClientModelClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                )
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(128, feature_dim)
                self.input_rows = input_rows
            
            def forward(self, x):
                if x.dim() == 2:
                    x = x.view(x.shape[0], 1, self.input_rows, -1)
                elif x.dim() == 3:
                    x = x.unsqueeze(1)
                x = self.features(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        class ServerModelClass(nn.Module):
            def __init__(self):
                super().__init__()
                total_feature_dim = feature_dim * n_clients
                self.fc1 = nn.Linear(total_feature_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, num_classes)
            
            def forward(self, *client_features):
                x = torch.cat(client_features, dim=1)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                # 添加 LogSoftmax 以配合 NLLLoss
                return F.log_softmax(x, dim=1)
                # 添加 LogSoftmax 以配合 NLLLoss
                return F.log_softmax(x, dim=1)
        
        return ClientModelClass, ServerModelClass


if __name__ == '__main__':
    # 测试单客户端版本
    print("=== 单客户端版本测试 ===")
    client_model, server_model = SplitResNet18.create_single_client_models()
    
    x = torch.randn(64, 784)  # 模拟输入
    features = client_model(x)
    output = server_model(features)
    
    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    
    # 测试多客户端版本
    print("\n=== 多客户端版本测试 ===")
    client_model, server_model = SplitResNet18.create_multi_client_models(n_clients=4)
    
    # 模拟 4 个客户端的输入
    client_inputs = [torch.randn(64, 196) for _ in range(4)]  # 每个客户端 196=7*28
    
    client_features = [client_model(inp) for inp in client_inputs]
    output = server_model(*client_features)
    
    print(f"Each client input shape: {client_inputs[0].shape}")
    print(f"Each client features shape: {client_features[0].shape}")
    print(f"Output shape: {output.shape}")
