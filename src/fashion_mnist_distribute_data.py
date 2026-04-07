"""
Fashion-MNIST 数据分发器

将 Fashion-MNIST 图像垂直分割并分发到多个客户端
图像尺寸: 28x28 灰度图 (与 MNIST 相同)
"""

import random

DATA_GENERATION_PROBABILITY = 0.2
TEST_SET_SIZE = 50
TESTING_GEN_SEED = 10
ESTIMATION_DATA_GENERATION_PROBABILITY = 0.02


class DiscreteDistributeFashionMNIST:
    """
    Fashion-MNIST 数据垂直分发类
    
    将每张 28x28 图像按行切分给多个客户端
    每个客户端收到图像的一部分行
    
    Example:
        >>> from torchvision import datasets, transforms
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> trainset = datasets.FashionMNIST('fashion_mnist', download=True, train=True, transform=transform)
        >>> trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        >>> 
        >>> distributed_data = DiscreteDistributeFashionMNIST(
        ...     data_owners=(client_1, client_2, client_3, client_4),
        ...     data_loader=trainloader
        ... )
        >>> 
        >>> # 每个客户端收到 (batch, 1, 7, 28) 的图像部分
        >>> distributed_data.data_pointer[0]['client_1'].shape
        torch.Size([64, 1, 7, 28])
    """
    
    def __init__(self, data_owners, data_loader):
        """
        Args:
            data_owners: tuple of data owners (VirtualWorker)
            data_loader: torch.utils.data.DataLoader for Fashion-MNIST
        """
        random.seed(TESTING_GEN_SEED)

        self.data_owners = data_owners
        self.data_loader = data_loader
        self.no_of_owner = len(data_owners)

        self.data_pointer = []
        """
        self.data_pointer: list of dictionaries
        (key, value) = (id of the data holder, pointer to the batch at that holder)
        
        Example:
        self.data_pointer = [
            {"client_1": pointer_to_client1_batch1, "client_2": pointer_to_client2_batch1, ...},
            {"client_1": pointer_to_client1_batch2, "client_2": pointer_to_client2_batch2, ...},
            ...
        ]
        """

        self.labels = []
        self.distributed_subdata = []
        self.test_set = []

        # 遍历每个 batch，切分图像并发送到客户端
        for images, labels in self.data_loader:
            curr_data_dict = {}

            # 按行均匀分割图像
            # Fashion-MNIST: 28x28, 分成 n_clients 份
            height = images.shape[-1] // self.no_of_owner

            self.labels.append(labels)

            # 分发图像到除最后一个客户端之外的所有客户端
            for i, owner in enumerate(self.data_owners[:-1]):
                # 切分图像并发送到 VirtualWorker
                image_part_ptr = images[:, :, :, height * i : height * (i + 1)].send(owner)
                curr_data_dict[owner.id] = image_part_ptr

            # 最后一个客户端接收剩余部分
            last_owner = self.data_owners[-1]
            last_part_ptr = images[:, :, :, height * (i + 1) :].send(last_owner)
            curr_data_dict[last_owner.id] = last_part_ptr

            self.data_pointer.append(curr_data_dict)
        
        # 创建测试集
        for _ in range(TEST_SET_SIZE):
            idx = random.random() * len(self.data_pointer)
            idx = int(idx)
            self.test_set.append((self.data_pointer[idx], self.labels[idx]))
            self.data_pointer.pop(idx)
            self.labels.pop(idx)
    
    def __iter__(self):
        id = 0
        for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):
            yield (id, data_ptr, label)
            id += 1
    
    def __len__(self):
        return len(self.data_loader) - 1
    
    def generate_subdata(self):
        """生成训练子集"""
        self.distributed_subdata = []
        for id, data_ptr, target in self:
            if random.random() <= DATA_GENERATION_PROBABILITY:
                self.distributed_subdata.append((id, data_ptr, target))

    def generate_estimate_subdata(self):
        """生成用于互信息估计的子集"""
        est_subdata = []
        for id, data_ptr, target in self.distributed_subdata:
            if random.random() <= ESTIMATION_DATA_GENERATION_PROBABILITY:
                est_subdata.append((id, data_ptr, target))
        return est_subdata

    def split_samples_by_class(self, subdata):
        """按类别分组样本"""
        class_data = {}
        for id, data_ptr, target in subdata:
            # 注意：不使用 .item()，与原始代码保持一致
            if not target in class_data:
                class_data[target] = []
            class_data[target].append((data_ptr, id))
        return class_data


# 为了兼容性，创建别名
DistributeFashionMNIST = DiscreteDistributeFashionMNIST


if __name__ == '__main__':
    # 测试代码
    import torch
    from torchvision import datasets, transforms
    import syft as sy
    
    # 初始化 PySyft
    hook = sy.TorchHook(torch)
    
    # 创建虚拟工作节点
    client_1 = sy.VirtualWorker(hook, id="client_1")
    client_2 = sy.VirtualWorker(hook, id="client_2")
    client_3 = sy.VirtualWorker(hook, id="client_3")
    client_4 = sy.VirtualWorker(hook, id="client_4")
    
    # 加载 Fashion-MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.FashionMNIST('fashion_mnist', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    # 分发数据
    distributed_data = DiscreteDistributeFashionMNIST(
        data_owners=(client_1, client_2, client_3, client_4),
        data_loader=trainloader
    )
    
    print(f"Total batches: {len(distributed_data)}")
    print(f"Test set size: {len(distributed_data.test_set)}")
    
    # 检查第一个 batch
    first_batch = distributed_data.data_pointer[0]
    for client_id, ptr in first_batch.items():
        print(f"{client_id}: shape = {ptr.shape}")
