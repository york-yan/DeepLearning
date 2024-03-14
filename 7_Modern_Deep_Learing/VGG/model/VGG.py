import torch 
from torch import nn

def vgg_block(num_convs,in_channels,out_channels):
    layers=[]
    for _ in range (num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)) #添加数个卷积层
        layers.append(nn.ReLU())
        in_channels=out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks=[]
    in_channels=1
    for (num_convs,out_channels) in conv_arch:                         # 从conv_arch中获取卷积层的数量
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels)) # vgg 块
        in_channels=out_channels
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),                                                  # 计算一下展开后的形状
        nn.Linear(out_channels*7*7,4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,10)
        )

conv_arch=((1,64),(1,128),(2,256),(2,512),(2,512))
net=vgg(conv_arch)


if __name__ == '__main__':
    # print(net)
    X=torch.rand(size=(1,1,224,224))
    for blk in net:
        X=blk(X)
        print(blk.__class__.__name__,'output shape:\t',X.shape)
    print(net)