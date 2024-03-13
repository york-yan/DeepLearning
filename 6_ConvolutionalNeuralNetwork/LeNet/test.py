import torch
from models.LeNet import LeNet
from PIL import Image
from torchvision import transforms


model = LeNet()
model_weight=torch.load('/home/york/code/DeepLearning/6_ConvolutionalNeuralNetwork/LeNet/model9.pth')
import ipdb

model.load_state_dict(model_weight)

preprogress=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
    ])

def load_and_preprogress(image_path):
    image=Image.open(image_path)
    image.resize((28,28))
    image=preprogress(image).unsqueeze(0)
    return image
def evaluate_image(model,image):
    with torch.no_grad():
        model.eval()
        out=model(image)
        _,predicted=torch.max(out,1)
        return predicted.item()
if __name__ == '__main__':
    # 手动输入图像路径
    image_path = "/home/york/code/DeepLearning/6_ConvolutionalNeuralNetwork/下载.png"

    # 加载并预处理图像
    image = load_and_preprogress(image_path)

    # 使用模型进行评估
    prediction = evaluate_image(model, image)
    # ipdb.set_trace()

    print("预测结果为：", prediction)