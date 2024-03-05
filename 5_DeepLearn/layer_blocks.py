import torch

from torch import nn
from torch.nn import functional as F

net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,20),nn.ReLU())
X=torch.rand(2,20)
# print(net(X))

# 自定义块=======================================================================
class MLP(nn.Module): # 注意看此处是nn.Module
    def __init__(self):
      super().__init__()
      self.hidden=nn.Linear(20,256)
      self.out=nn.Linear(256,10)
    
    def forward(self,X):
      return self.out(F.relu(self.hidden(X)))

net2=MLP()
# print(net2(X))

# 顺序块========================================================================
class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)]=module

    def forward(self,X):
        for block in self._modules.values():
            X=block(X)
        return X
net3=MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
# print(net3(X))

# 前向传播中执行代码==========================================================
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight=torch.rand((20,20),requires_grad=False)
        self.linear=nn.Linear(20,20)
    def forward(self,X):
        X=self.linear(X)
        X=F.relu(torch.mm(X,self.rand_weight)+1)
        X=self.linear(X)
        while X.abs().sum()>1:
            X/=2
        return X.sum()
net4=FixedHiddenMLP()
# print(net4(X))

# 混合块==============================================================================
class NestMLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net=nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.linear=nn.Linear(32,16)
    
    def forward(self,X):
        return self.linear(self.net(X))
net5=nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
print("混合输出")
# print(net5(X))