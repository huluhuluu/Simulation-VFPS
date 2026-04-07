# 简化测试脚本
import sys
sys.path.append('./')

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import syft as sy
import random
import time

print("Step 1: Importing modules...")

# 导入自定义模块
from src.discrete_splitnn import DiscreteSplitNN
from src.fashion_mnist_distribute_data import DiscreteDistributeFashionMNIST
from src.models.split_resnet import SplitResNet18

print("Step 2: Initializing PySyft...")

# 初始化 PySyft
hook = sy.TorchHook(torch)

print("Step 3: Loading Fashion-MNIST...")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.FashionMNIST(
    root='./datasets/fashion_mnist', 
    download=True, 
    train=True, 
    transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

print(f"Training set size: {len(trainset)}")

print("Step 4: Creating virtual workers...")

# 创建虚拟工作节点
N_CLIENTS = 4
clients = [sy.VirtualWorker(hook, id=f"client_{i}") for i in range(N_CLIENTS)]
server = sy.VirtualWorker(hook, id="server")

data_owners = tuple(clients)
model_locations = clients + [server]

print(f"Created {len(clients)} clients and 1 server")

print("Step 5: Distributing data...")

# 分发数据
distributed_trainloader = DiscreteDistributeFashionMNIST(
    data_owners=data_owners, 
    data_loader=trainloader
)

print(f"Distributed data batches: {len(distributed_trainloader)}")

print("Step 6: Creating models...")

# 创建分割模型
torch.manual_seed(0)

input_rows = 28 // N_CLIENTS  # 7
feature_dim = 256

ClientModel, ServerModel = SplitResNet18.create_multi_client_models(
    n_clients=N_CLIENTS,
    input_rows=input_rows,
    feature_dim=feature_dim,
    hidden_dim=128,
    num_classes=10
)

models = {f"client_{i}": ClientModel() for i in range(N_CLIENTS)}
models["server"] = ServerModel()

print("Step 7: Creating optimizers...")

# 创建优化器
optimizers = [
    (optim.SGD(models[loc.id].parameters(), lr=0.01, momentum=0.9), loc)
    for loc in model_locations
]

# 发送模型到工作节点
for loc in model_locations:
    models[loc.id].send(loc)

print("Step 8: Creating SplitNN...")

splitNN = DiscreteSplitNN(
    models=models,
    server=server,
    data_owners=clients,
    optimizers=optimizers,
    dist_data=distributed_trainloader,
    k=3,
    n_selected=2,
    padding_method="zeros"
)

print("Step 9: Generating training subdata...")

distributed_trainloader.generate_subdata()
print(f"Training subdata size: {len(distributed_trainloader.distributed_subdata)}")

print("Step 10: Running group testing (this may take a while)...")

# 减少测试次数
splitNN.group_testing(n_tests=3)

print("Group testing done!")
print("Selected clients:")
for client_id, selected in splitNN.selected.items():
    if client_id != "server":
        print(f"  {client_id}: {'Selected' if selected else 'Not selected'}")

print("Step 11: Training for 3 epochs...")

# 评估函数
def evaluate(splitnn, test_set):
    correct = 0
    total = 0
    
    for data_ptr, label in test_set:
        label = label.send(splitnn.server)
        pred = splitnn.predict(data_ptr)
        
        pred_np = pred.get().argmax(dim=1).numpy()
        label_np = label.get().numpy()
        
        correct += (pred_np == label_np).sum()
        total += len(label_np)
    
    return correct / total

# 训练循环
total_start_time = time.time()

for epoch in range(3):
    epoch_start_time = time.time()
    
    running_loss = 0.0
    
    count = 0
    for _, data_ptr, label in distributed_trainloader.distributed_subdata:
        label = label.send(server)
        loss = splitNN.train(data_ptr, label)
        running_loss += loss
        count += 1
        if count >= 10:  # 只训练 10 个批次
            break
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = running_loss / count
    print(f"Epoch {epoch+1}/3 - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

total_time = time.time() - total_start_time

# 测试
print("\n--- Evaluating ---")
eval_start = time.time()
accuracy = evaluate(splitNN, distributed_trainloader.test_set)
eval_time = time.time() - eval_start
print(f"Test Accuracy: {accuracy*100:.2f}% - Eval Time: {eval_time:.2f}s")
print(f"Total Training Time: {total_time:.2f}s")
print("\nTest completed successfully!")