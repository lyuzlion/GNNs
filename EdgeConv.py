import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch.nn import Linear
from sklearn.utils import shuffle
import numpy as np

import warnings
warnings.filterwarnings("ignore")
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))
        # 构建拥有两个线性层的MLP，层间的激活函数是ReLU
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        # 为什么dim=1？因为是按照第1维拼接（维数从0开始数）
        return self.mlp(tmp)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = EdgeConv(dataset.num_node_features, 16)
        self.conv2 = EdgeConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # Dropout 是一种正则化技术，用于防止神经网络在训练过程中出现过拟合
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'], pre_transform=T.KNNGraph(k=6))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
model = GCN().to(device)
data = dataset[0]

def sample_mask(idx, l):
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)

seed = 2
shuffled_idx = shuffle(np.array(range(data.num_nodes)), random_state=seed) # 已经被随机打乱
train_idx = shuffled_idx[:int(0.7 * data.num_nodes)].tolist()
val_idx = shuffled_idx[int(0.7 * data.num_nodes): int(0.9 * data.num_nodes)].tolist()
test_idx = shuffled_idx[int(0.9 * data.num_nodes):].tolist()
train_mask = sample_mask(train_idx, data.num_nodes)
val_mask = sample_mask(val_idx, data.num_nodes)
test_mask = sample_mask(test_idx, data.num_nodes)

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train() # 打开训练模式，启用 batch normalization 和 dropout
for epoch in range(200):
    optimizer.zero_grad() # 梯度置零
    out = model(data) # 获得一个二维矩阵，代表点i属于第j类的概率
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    # print(loss)
    loss.backward()
    optimizer.step()

model.eval() # 打开评估模式，不启用 Batch Normalization 和 Dropout
pred = model(data).argmax(dim=1)
correct = (pred[test_mask] == data.y[test_mask]).sum()
acc = int(correct) / int(test_mask.sum())
print(f'Accuracy: {acc:.4f}')