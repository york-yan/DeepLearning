import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
# print(net(X))
# print (net[1].state_dict())
# print(net[2].state_dict())
# 通过索引访问模型的任意层参数
print(*[name for name, param in net[0].named_parameters()])