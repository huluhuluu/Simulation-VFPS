# Dynamic-VFPS 代码逻辑解释

## 一、项目概述

本项目实现了论文 **"VF-PS: How to Select Important Participants in Vertical Federated Learning, Efficiently and Securely?"** 的核心算法，用于在垂直联邦学习中高效且安全地选择重要参与者。

---

## 二、核心概念

### 2.1 垂直联邦学习 (Vertical Federated Learning, VFL)

**场景**: 多个客户端拥有同一批样本的不同特征（垂直数据划分）

```
样本1: [客户端1: 特征A] + [客户端2: 特征B] + [客户端3: 特征C] → 标签
样本2: [客户端1: 特征A] + [客户端2: 特征B] + [客户端3: 特征C] → 标签
...
```

**本项目实现**:
- Fashion-MNIST 图像 (28x28) 垂直切分为 N_CLIENTS 份
- 每个客户端拥有图像的一部分行
- 例如: 10个客户端，每个客户端拥有 2.8 行 ≈ 2-3 行

### 2.2 动态参与者选择

**问题**: 不是所有客户端的贡献都相同，如何选择最重要的参与者？

**解决方案**: 基于**互信息 (Mutual Information)** 估计每个参与者对模型性能的贡献

---

## 三、核心算法流程

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Dynamic-VFPS 流程                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  数据分发      │       │  模型训练      │       │  参与者选择    │
│  (Vertical)   │       │  (SplitNN)    │       │  (MI-based)   │
└───────────────┘       └───────────────┘       └───────────────┘
```

### 3.2 详细流程

#### 阶段1: 初始化

```python
# 1. 创建虚拟工作节点
clients = [sy.VirtualWorker(hook, id=f"client_{i}") for i in range(N_CLIENTS)]
server = sy.VirtualWorker(hook, id="server")

# 2. 数据垂直切分
distributed_trainloader = DiscreteDistributeFashionMNIST(
    data_owners=clients, 
    data_loader=trainloader
)
# 每张 28x28 图像被切分为 N_CLIENTS 份
# client_0: 行 0-2, client_1: 行 3-5, ...
```

#### 阶段2: 组测试 (Group Testing)

**目的**: 估计每个参与者的互信息贡献

```python
def group_testing(n_tests):
    """
    核心算法：通过随机组测试估计参与者贡献
    
    原理：
    1. 随机选择一组参与者组合
    2. 计算该组合的互信息 (MI)
    3. 累计每个参与者的分数
    4. 选择 top-k 参与者
    """
    scores = {}
    for _ in range(n_tests):
        # 随机生成测试组
        test_instance = random_select_clients(p=0.5)
        
        # 计算该组的互信息
        mi = knn_mi_estimator(test_instance)
        
        # 累加分数
        for client in test_instance:
            scores[client] += mi
    
    # 选择分数最高的 top-k 客户端
    selected = top_k(scores, n_selected)
```

#### 阶段3: KNN 互信息估计

```python
def knn_mi_estimator(distributed_subdata):
    """
    使用 KNN 方法估计互信息
    
    公式: MI(X, Y) = ψ(N) - ψ(Nq) + ψ(k) - ψ(mq)
    
    其中:
    - ψ: digamma 函数
    - N: 总样本数
    - Nq: 同类别样本数
    - k: KNN 参数
    - mq: 近邻数
    """
    # 1. 计算各客户端间的距离
    for client in clients:
        distances[client] = torch.cdist(data[client], data2[client])
    
    # 2. 聚合距离
    aggregate_distances = sum(distances)
    
    # 3. 计算 MI
    mi = digamma(N) - digamma(Nq) + digamma(k) - digamma(mq)
    return mi
```

#### 阶段4: 训练循环

```python
for epoch in range(EPOCHS):
    # 1. 动态更新参与者选择
    if random() < SUBSET_UPDATE_PROB:
        generate_subdata()          # 更新训练子集
        group_testing(n_tests)      # 重新选择参与者
    
    # 2. 前向传播 (仅选中参与者)
    for client in clients:
        if selected[client]:
            output = model[client](data[client])
        else:
            output = padding_zeros()  # 未选中用零填充
    
    # 3. 服务器聚合
    server_input = concat(all_outputs)
    pred = server_model(server_input)
    
    # 4. 反向传播 (仅更新选中参与者)
    loss.backward()
    for client in selected_clients:
        optimizer[client].step()
```

---

## 四、关键参数说明

### 4.1 客户端数量与选择比例

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `N_CLIENTS` | 10 | 客户端总数 |
| `N_SELECTED` | 5-7 | 每轮选择的客户端数量 |
| **参与率** | 50%-70% | N_SELECTED / N_CLIENTS |

**选择比例建议**:
- **过低 (< 30%)**: 模型性能下降，信息不足
- **适中 (50%-70%)**: 平衡效率与性能 ✓
- **过高 (> 80%)**: 失去动态选择的意义

### 4.2 其他关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_tests` | 5-10 | 组测试次数，越多越准确但越慢 |
| `k` (KNN) | 3 | KNN 近邻参数 |
| `SUBSET_UPDATE_PROB` | 0.2 | 每轮更新子集的概率 |
| `PADDING_METHOD` | "zeros" | 未选中客户端的填充方式 |

---

## 五、数据流图

```
Fashion-MNIST Dataset (60000 张 28x28 图像)
           │
           ▼
    DataLoader (batch_size=64)
           │
           ▼
┌──────────────────────────────────────┐
│     DiscreteDistributeFashionMNIST    │
│  (垂直切分 + 分发到各客户端)            │
└──────────────────────────────────────┘
           │
           ├──── client_0: (batch, 1, 2, 28) ────┐
           ├──── client_1: (batch, 1, 3, 28) ────┤
           ├──── client_2: (batch, 1, 3, 28) ────┤
           ├──── ...                            │
           └──── client_9: (batch, 1, 3, 28) ────┤
                                              │
           ┌──────────────────────────────────┘
           ▼
┌──────────────────────────────────────┐
│          Group Testing               │
│  (估计每个客户端的 MI 分数)            │
└──────────────────────────────────────┘
           │
           ▼
    Selected: [client_0, client_2, client_5, ...]
           │
           ▼
┌──────────────────────────────────────┐
│           Training                   │
│  - Selected: 计算特征                │
│  - Not selected: Padding (zeros)     │
└──────────────────────────────────────┘
           │
           ▼
    Server: Concat + Classification
           │
           ▼
        Loss + Backprop
           │
           ▼
    Update selected clients only
```

---

## 六、模型架构

### 6.1 客户端模型

```python
# 每个客户端的 CNN 特征提取器
ClientModel:
    Conv2d(1, 32, 3x3) → BatchNorm → ReLU
    Conv2d(32, 64, 3x3) → BatchNorm → ReLU
    Conv2d(64, 128, 3x3) → BatchNorm → ReLU
    AdaptiveAvgPool2d(1, 1)
    Linear(128, 256)  # 输出 256 维特征向量
```

### 6.2 服务器模型

```python
# 服务器聚合分类器
ServerModel:
    Input: concat(256 * N_SELECTED) = 1536 维 (N_SELECTED=6)
    Linear(1536, 128) → ReLU
    Linear(128, 10)
    LogSoftmax(dim=1)  # 配合 NLLLoss
```

---

## 七、与论文的对应关系

| 论文内容 | 代码实现 |
|---------|---------|
| 垂直数据划分 | `DiscreteDistributeFashionMNIST` |
| 组测试 | `group_testing()` |
| KNN 互信息估计 | `knn_mi_estimator()` |
| 动态参与者选择 | `selected` 字典 + `n_selected` 参数 |
| Padding 机制 | `generate_data()` + `PADDING_METHOD` |
| 仅更新选中参与者 | `optimizer.step()` 条件执行 |

---

## 八、运行示例

```bash
# 在 fvps 环境中运行
conda activate fvps
python test.py
```

### 预期输出

```
Training set size: 60000
Created 10 clients and 1 server
Distributed data batches: 937
Client model parameters: 126,144
Server model parameters: 165,130

Initial participant selection:
  client_0: Selected
  client_1: Selected
  client_2: Selected
  client_3: Selected
  client_4: Selected
  client_5: Selected
  client_6: Not selected
  client_7: Not selected
  client_8: Not selected
  client_9: Not selected

Epoch 1/50 - Loss: 2.2840 - Time: 1.64s
Epoch 2/50 - Loss: 2.1982 - Time: 1.56s
...

--- Evaluating at Epoch 10 ---
Test Accuracy: 42.15% - Eval Time: 4.83s

...

Final Test Accuracy: 65.32%
Total Training Time: 82.45s (1.37 min)
```

---

## 九、常见问题

### Q1: 为什么 Loss 是负数？

**已修复**: Server 模型需要输出 `LogSoftmax` 才能配合 `NLLLoss`

### Q2: 如何选择 `N_SELECTED`?

**推荐**: 50%-70% 的客户端参与率
- N_CLIENTS=10 → N_SELECTED=5-7
- 理由: 平衡训练效率和模型性能

### Q3: 组测试次数 `n_tests` 如何设置？

**推荐**: 5-10 次
- 过少: 分数估计不准确
- 过多: 计算开销大

### Q4: 为什么有些客户端从未被选中？

**可能原因**:
1. 该客户端的特征对分类贡献较小（MI 分数低）
2. `n_tests` 太少，估计不准确
3. 数据切分方式导致某些客户端特征冗余

---

## 十、扩展方向

1. **更复杂的模型**: 替换为完整 ResNet18
2. **真实网络延迟**: 模拟真实联邦学习场景
3. **隐私保护**: 添加差分隐私或安全聚合
4. **异步训练**: 支持客户端异步更新
5. **更多数据集**: 支持 CIFAR-10、Medical 数据集等

---

## 参考

- 论文: [VF-PS: How to Select Important Participants in Vertical Federated Learning](https://openreview.net/pdf?id=vNrSXIFJ9wz)
- PySyft: [https://github.com/OpenMined/PySyft](https://github.com/OpenMined/PySyft)
