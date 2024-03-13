import torch 
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.Sigmoid())
        
        self.avgpool1=nn.AvgPool2d((2,2),stride=2)

        self.conv2=nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5),
            nn.Sigmoid()
        )

        self.avgpool2=nn.AvgPool2d((2,2),stride=2)
        self.fc1=nn.Sequential(nn.Linear(16*5*5,120),nn.Sigmoid())
        self.fc2=nn.Sequential(nn.Linear(120,84),nn.Sigmoid())
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        x=self.conv1(x)
        x=self.avgpool1(x)
        x=self.conv2(x)
        x=self.avgpool2(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x
    
if __name__=='__main__':
    net=LeNet()
    print(net)
