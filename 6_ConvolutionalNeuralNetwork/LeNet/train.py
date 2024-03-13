import os
import sys
sys.path.append('/home/york/code/DeepLearning/6_ConvolutionalNeuralNetwork/LeNet')
import torch
from torchvision import datasets,transforms
from models.LeNet import LeNet
import ipdb

if __name__=='__main__':
    net=LeNet()
    net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # print(net)
    BATCH_SIZE=16
    EPOCHS=20
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer=torch.optim.Adam(net.parameters())

    train_loader=torch.utils.data.DataLoader(
        datasets.MNIST('data',train=True,download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,),(0.3081,))
                       ])),batch_size=BATCH_SIZE,shuffle=True
    )
    test_loader=torch.utils.data.DataLoader(
        datasets.MNIST('data',train=False,download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,),(0.3081,))
                       ])),batch_size=BATCH_SIZE,shuffle=True
    )
    for epoch in range (EPOCHS):
        net.to(DEVICE)
        net.train()
        
        # ipdb.set_trace()
        for data,target in train_loader:
            data,target=data.to(DEVICE),target.to(DEVICE)
            output=net(data)
            loss=torch.nn.functional.cross_entropy(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        correct=0
        with torch.no_grad():
            for data,target in test_loader:
                data,target=data.to(DEVICE),target.to(DEVICE)
                output=net(data)
                pred=output.argmax(dim=1,keepdim=True)
                correct+=pred.eq(target.view_as(pred)).sum().item()
        print('Epoch:{}\tAccuracy:{:.2f}%'.format(epoch,correct/len(test_loader.dataset)*100))
        torch.save(net.state_dict(),'model{}.pth'.format(epoch))

