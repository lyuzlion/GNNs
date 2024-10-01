import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import warnings
warnings.filterwarnings("ignore")
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # 特征矩阵 x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: 向邻接矩阵增加自环，_没用
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: 把点的特征矩阵线性变换，也就是乘上参数矩阵
        x = self.lin(x)

        # Step 3: 归一化
        row, col = edge_index # 因为edge_index有两行，所以row是第一行(所有边的起点)，col是第二行(所有边的终点)
        deg = degree(row, x.size(0), dtype=x.dtype) # 获取每个点的点度
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] # norm[i]=deg_inv_sqrt[row[i]] * deg_inv_sqrt[col[i]]
        # norm是[1, |E|]的

        # Step 4-5: 聚合信息，propagate() 会自动调用message(), aggregate(), update()
        out = self.propagate(edge_index, x=x, norm=norm)
        # Step 6: 添加一个偏移量.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j # .view(-1, 1)表示转换为列向量



class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # Dropout 是一种正则化技术，用于防止神经网络在训练过程中出现过拟合
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
dataset = Planetoid(root='/tmp/Cora', name='Cora')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train() # 打开训练模式，启用 batch normalization 和 dropout
for epoch in range(200):
    optimizer.zero_grad() # 梯度置零
    out = model(data) # 获得一个二维矩阵，代表点i属于第j类的概率
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # print(out[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval() # 打开评估模式，不启用 Batch Normalization 和 Dropout
pred = model(data).argmax(dim=1) # 划为概率最大的那一类
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')