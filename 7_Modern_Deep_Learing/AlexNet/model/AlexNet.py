# import torch
# from torch import nn

# class AlexNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features=nn.Sequential(
#             nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3,stride=2),
#             nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3,stride=2),
#             nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3,stride=2),
#             nn.Flatten(),
#             nn.Linear(6400,4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096,4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096,10)
#         )
#     def forward(self,x):
#         return self.features(x)
    
# if __name__ == '__main__':
#     alexnet = AlexNet()
#     print(alexnet)
    

import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6400, 4096)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.features(x)
        print("Features output size:", x.shape)

        x = self.flatten(x)
        print("Flatten output size:", x.shape)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        print("FC1 output size:", x.shape)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        print("FC2 output size:", x.shape)

        x = self.fc3(x)
        print("FC3 output size:", x.shape)

        return x

if __name__ == '__main__':
    alexnet = AlexNet()
    # print(alexnet)
    print(alexnet(torch.randn(1, 1, 224, 224)))
