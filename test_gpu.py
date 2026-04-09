#!/usr/bin/env python3
"""
Dynamic-VFPS GPU Test Script
MI-based Dynamic Participant Selection for Vertical Federated Learning

Usage:
    python test_gpu.py                                    # default parameters
    python test_gpu.py --epochs 50 --clients 10          # custom parameters
    python test_gpu.py --encryption paillier             # use Paillier encryption
    python test_gpu.py --mi-mode static                  # static client selection
    python test_gpu.py --help                            # show all parameters
"""

import sys
sys.path.append('./')

import random
import torch
from torchvision import datasets, transforms
from torch import optim

# Import refactored modules
from src.config import Config
from src.utils.helpers import parse_args, get_device
from src.communication.estimator import CommunicationEstimator
from src.data.distributor import DataDistributor
from src.models.split_resnet import SplitResNet18
from src.splitnn import SplitNN
from src.evaluation import evaluate


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    config = Config.from_args(args)
    
    # 设备
    device = get_device()
    if torch.cuda.is_available():
        print(f"[INFO] Device: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print(f"[INFO] Device: CPU")
    
    # 打印配置
    print(f"\n{'='*60}")
    print(f"Configuration: {config}")
    print(f"{'='*60}\n")
    
    # 创建通信估算器
    comm_estimator = CommunicationEstimator(
        bandwidth_mbps=config.bandwidth_mbps,
        encryption=config.encryption
    )
    
    # 加载数据
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    
    # 加载真正的测试集
    testset = datasets.FashionMNIST(
        root='./datasets/fashion_mnist',
        download=True,
        train=False,  # 使用测试数据
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Dataset: Fashion-MNIST")
    print(f"  Train: {len(trainset)} samples, {len(trainloader)} batches")
    print(f"  Test:  {len(testset)} samples, {len(testloader)} batches")
    
    # 数据分发
    distributor = DataDistributor(config.n_clients, trainloader, device, testloader)
    print(f"Data distributed: {distributor.n_batches} batches, {config.n_clients} clients")
    
    # 创建模型
    torch.manual_seed(0)
    input_width = 28 // config.n_clients  # 每个客户端的图像宽度
    
    ClientModel, ServerModel = SplitResNet18.create_multi_client_models(
        n_clients=config.n_clients,
        input_width=input_width,
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes
    )
    
    models = {f"client_{i}": ClientModel().to(device) for i in range(config.n_clients)}
    models["server"] = ServerModel().to(device)
    
    client_params = sum(p.numel() for p in models["client_0"].parameters())
    server_params = sum(p.numel() for p in models["server"].parameters())
    print(f"Model: Client {client_params:,} params, Server {server_params:,} params")
    
    # 创建优化器
    optimizers = {f"client_{i}": optim.SGD(models[f"client_{i}"].parameters(), 
                                           lr=config.learning_rate, momentum=0.9)
                  for i in range(config.n_clients)}
    optimizers["server"] = optim.SGD(models["server"].parameters(), 
                                     lr=config.learning_rate, momentum=0.9)
    
    # 创建 SplitNN
    splitnn = SplitNN(models, config, optimizers, comm_estimator, device)
    
    # -------------------------------------------------------------------------
    # 训练循环
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[Training Started]")
    print(f"{'='*60}")
    print(f"MI Mode: {args.mi_mode}")
    
    total_train_time = 0.0
    total_mi_compute_time = 0.0  # MI 计算时间
    total_mi_comm_time = 0.0     # MI 通信时间
    total_comm_time = 0.0        # 模型训练通信时间
    global_step = 0
    
    # 打印数据维度信息（只在第一个 step 打印一次）
    dim_info_printed = False
    
    # =========================================================================
    # Static 模式：训练前一次性选择客户端
    # =========================================================================
    if args.mi_mode == 'static':
        print(f"\n[Static MI Mode]")
        print(f"  MI data ratio: {args.mi_ratio}")
        
        # 1. 从训练集取 mi_ratio 比例的数据用于 MI 计算
        n_mi_batches = int(len(distributor.data_pointer) * args.mi_ratio)
        mi_indices = random.sample(range(len(distributor.data_pointer)), n_mi_batches)
        mi_data = [(idx, distributor.data_pointer[idx], distributor.labels[idx]) 
                   for idx in mi_indices]
        print(f"  MI batches: {n_mi_batches}")
        
        # 2. 用所有 MI 数据进行组测试选择客户端
        scores, mi_comm_time, mi_compute_time = splitnn.group_testing(mi_data, config.n_tests)
        
        selected = [k for k, v in splitnn.selected.items() if v]
        print(f"  Selected clients: {selected}")
        print(f"  MI total time: {mi_compute_time + mi_comm_time:.2f}s")
        print(f"  MI compute time: {mi_compute_time:.2f}s")
        print(f"  MI comm time: {mi_comm_time:.4f}s ({n_mi_batches} batches)")
        print(f"\n  Clients fixed for all training steps")
        
        total_mi_compute_time = mi_compute_time
        total_mi_comm_time = mi_comm_time
    
    # =========================================================================
    # 训练循环
    # =========================================================================
    for epoch in range(config.epochs):
        # 生成本轮训练数据
        distributor.generate_subdata(config.subset_update_prob)
        
        print(f"\n[Epoch {epoch+1}/{config.epochs}]")
        
        epoch_train_time = 0.0
        epoch_comm_time = 0.0
        
        # 训练
        for _, data_ptr, label in distributor.subdata:
            # Dynamic 模式：每个 step 前进行客户端选择
            if args.mi_mode == 'dynamic':
                estimate_data = distributor.generate_estimate_subdata(config.estimate_samples)
                scores, mi_comm_time, mi_compute_time = splitnn.group_testing(estimate_data, config.n_tests)
                
                total_mi_compute_time += mi_compute_time
                total_mi_comm_time += mi_comm_time
                
                selected = [k for k, v in splitnn.selected.items() if v]
            
            label = label.to(device)
            
            # 打印维度信息（只打印一次）
            if not dim_info_printed:
                print(f"\n{'='*60}")
                print("[Data Dimensions]")
                print(f"{'='*60}")
                for i in range(config.n_clients):
                    client_id = f"client_{i}"
                    input_shape = data_ptr[client_id].shape
                    input_size = input_shape[0] * input_shape[1]
                    print(f"  {client_id} input: {tuple(input_shape)} = {input_size} elements")
                print(f"\n  [After client forward (activation to transmit)]")
                for i in range(config.n_clients):
                    client_id = f"client_{i}"
                    with torch.no_grad():
                        output = splitnn.models[client_id](data_ptr[client_id])
                        output_shape = output.shape
                        output_size = output_shape[0] * output_shape[1]
                        print(f"  {client_id} activation: {tuple(output_shape)} = {output_size} elements")
                print(f"\n  [Server receives from {len(selected)} selected clients]")
                total_activation_size = 0
                for client_id in selected:
                    with torch.no_grad():
                        output = splitnn.models[client_id](data_ptr[client_id])
                        total_activation_size += output.numel()
                print(f"  Total activation size: {total_activation_size} elements = {total_activation_size * 4 / 1024:.2f} KB")
                print(f"{'='*60}\n")
                dim_info_printed = True
            
            loss, train_time, comm_time = splitnn.train_step(data_ptr, label, config.local_epochs)
            
            global_step += 1
            epoch_train_time += train_time
            epoch_comm_time += comm_time
            total_train_time += train_time
            total_comm_time += comm_time
            
            # 定期评估
            if global_step % config.eval_every_steps == 0:
                acc = evaluate(splitnn, distributor.test_set[:10], device)
                
                # 整体累计时间
                overall_total_time = (total_train_time + total_comm_time + 
                                     total_mi_compute_time + total_mi_comm_time)
                
                print(f"  Step {global_step:4d} | Selected: {selected} | Loss: {loss:.4f} | Acc: {acc*100:5.2f}%")
                
                # Dynamic 模式显示每步的 MI 时间
                if args.mi_mode == 'dynamic':
                    print(f"         Step: {train_time + comm_time + mi_compute_time + mi_comm_time:.3f}s")
                    print(f"           - Train: {train_time:.3f}s")
                    print(f"           - Comm: {comm_time:.4f}s")
                    print(f"           - MI Compute: {mi_compute_time:.3f}s")
                    print(f"           - MI Comm: {mi_comm_time:.4f}s")
                else:
                    print(f"         Step: {train_time + comm_time:.3f}s")
                    print(f"           - Train: {train_time:.3f}s")
                    print(f"           - Comm: {comm_time:.4f}s")
                
                print(f"         Total: {overall_total_time:.2f}s")
                print(f"           - Train: {total_train_time:.2f}s")
                print(f"           - Comm: {total_comm_time:.2f}s")
                print(f"           - MI Compute: {total_mi_compute_time:.2f}s")
                print(f"           - MI Comm: {total_mi_comm_time:.2f}s")
    
    # -------------------------------------------------------------------------
    # 最终评估
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[Final Evaluation]")
    print(f"{'='*60}")
    
    final_acc = evaluate(splitnn, distributor.test_set, device)
    total_time = (total_train_time + total_comm_time + 
                 total_mi_compute_time + total_mi_comm_time)
    
    print(f"Accuracy: {final_acc*100:.2f}%")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  - Training Time: {total_train_time:.2f}s")
    print(f"  - Communication Time: {total_comm_time:.2f}s")
    print(f"  - MI Compute Time: {total_mi_compute_time:.2f}s")
    print(f"  - MI Comm Time: {total_mi_comm_time:.2f}s")
    print(f"Data Transferred: {comm_estimator.total_data_mb:.2f} MB")
    print(f"Encryption: {config.encryption}")
    print(f"MI Mode: {args.mi_mode}")
    print(f"Local Epochs: {config.local_epochs}")
    print(f"Total Steps: {global_step}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()