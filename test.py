# 配置参数
EPOCHS = 50
SUBSET_UPDATE_PROB = 0.2
PADDING_METHOD = "zeros"
LEARNING_RATE = 0.001
BATCH_SIZE = 256
N_CLIENTS = 10   # 客户端总数
N_SELECTED = 6   # 每轮选择的客户端数量 (60%参与率)
N_TESTS = 5      # 组测试次数

import sys
sys.path.append('./')

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import syft as sy
import random
import time

# 导入自定义模块
from src.discrete_splitnn import DiscreteSplitNN
from src.fashion_mnist_distribute_data import DiscreteDistributeFashionMNIST
from src.models.split_resnet import SplitResNet18, MultiClientNet, MultiClientServerNet

# 初始化 PySyft
hook = sy.TorchHook(torch)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 Fashion-MNIST
trainset = datasets.FashionMNIST(
    root='./datasets/fashion_mnist', 
    download=True, 
    train=True, 
    transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training set size: {len(trainset)}")
print(f"Number of batches: {len(trainloader)}")

# 创建虚拟工作节点
clients = [sy.VirtualWorker(hook, id=f"client_{i}") for i in range(N_CLIENTS)]
server = sy.VirtualWorker(hook, id="server")

data_owners = tuple(clients)
model_locations = clients + [server]

print(f"Created {len(clients)} clients and 1 server")

# 分发数据
distributed_trainloader = DiscreteDistributeFashionMNIST(
    data_owners=data_owners, 
    data_loader=trainloader
)

print(f"Distributed data batches: {len(distributed_trainloader)}")

# 创建分割模型
torch.manual_seed(0)

input_rows = 28 // N_CLIENTS
feature_dim = 256

# 创建模型实例
ClientModel, ServerModel = SplitResNet18.create_multi_client_models(
    n_clients=N_CLIENTS,
    input_rows=input_rows,
    feature_dim=feature_dim,
    hidden_dim=128,
    num_classes=10
)

# 为每个客户端创建模型
models = {f"client_{i}": ClientModel() for i in range(N_CLIENTS)}
models["server"] = ServerModel()

# 打印模型信息
total_params = sum(p.numel() for p in models["client_0"].parameters())
server_params = sum(p.numel() for p in models["server"].parameters())
print(f"Client model parameters: {total_params:,}")
print(f"Server model parameters: {server_params:,}")

# 创建优化器并发送模型到工作节点
optimizers = [
    (optim.SGD(models[loc.id].parameters(), lr=LEARNING_RATE, momentum=0.9), loc)
    for loc in model_locations
]

# 发送模型到对应的工作节点
for loc in model_locations:
    models[loc.id].send(loc)

print("Models sent to workers")

# 创建 SplitNN 实例
splitNN = DiscreteSplitNN(
    models=models,
    server=server,
    data_owners=clients,
    optimizers=optimizers,
    dist_data=distributed_trainloader,
    k=3,  # KNN 参数
    n_selected=N_SELECTED,
    padding_method=PADDING_METHOD
)

print(f"SplitNN created with k={splitNN.k}, n_selected={splitNN.n_selected}")

# 初始组测试
distributed_trainloader.generate_subdata()
print(f"Training subdata size: {len(distributed_trainloader.distributed_subdata)}")

splitNN.group_testing(n_tests=N_TESTS)

print("Initial participant selection:")
for client_id, selected in splitNN.selected.items():
    if client_id != "server":
        print(f"  {client_id}: {'Selected' if selected else 'Not selected'}")

print(f"\nTest set size: {len(distributed_trainloader.test_set)} batches")
print("=" * 60)

# 评估函数
def evaluate(splitnn, test_set):
    correct = 0
    total = 0
    
    for data_ptr, label in test_set:
        label = label.send(splitnn.server)
        pred = splitnn.predict(data_ptr)
        
        # 获取预测结果
        pred_np = pred.get().argmax(dim=1).numpy()
        label_np = label.get().numpy()
        
        correct += (pred_np == label_np).sum()
        total += len(label_np)
    
    return correct / total

# 训练循环
total_start_time = time.time()
total_train_time = 0.0
total_eval_time = 0.0
global_step = 0
EVAL_EVERY_STEPS = 10  # 每10个step测试一次

for epoch in range(EPOCHS):
    # 随机更新子集和参与者选择
    if random.random() < SUBSET_UPDATE_PROB:
        distributed_trainloader.generate_subdata()
        splitNN.group_testing(n_tests=N_TESTS)
    
    # 训练
    for _, data_ptr, label in distributed_trainloader.distributed_subdata:
        step_start_time = time.time()
        
        # 发送标签到服务器
        label = label.send(server)
        
        # 训练
        loss = splitNN.train(data_ptr, label)
        global_step += 1
        
        step_time = time.time() - step_start_time
        total_train_time += step_time
        
        # 每10个step测试一次并打印
        if global_step % EVAL_EVERY_STEPS == 0:
            eval_start = time.time()
            accuracy = evaluate(splitNN, distributed_trainloader.test_set)
            eval_time = time.time() - eval_start
            total_eval_time += eval_time
            
            print(f"Step {global_step} | Epoch {epoch+1} | Loss: {loss:.4f} | Train Time: {total_train_time:.2f}s | Test Acc: {accuracy*100:.2f}% | Eval Time: {eval_time:.2f}s")

total_time = time.time() - total_start_time

# 最终测试
print("\n" + "=" * 60)
print("Final Evaluation...")
accuracy = evaluate(splitNN, distributed_trainloader.test_set)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
print(f"Total Training Time: {total_train_time:.2f}s ({total_train_time/60:.2f} min)")
print(f"Total Eval Time: {total_eval_time:.2f}s ({total_eval_time/60:.2f} min)")
print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
print(f"Total Steps: {global_step}")
print("=" * 60)
print(f"Total Steps: {global_step}")
print("=" * 60)