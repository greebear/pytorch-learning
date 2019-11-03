import torchvision
import torch.nn as nn
import torch
from collections import OrderedDict

alexnet_ori = torchvision.models.AlexNet()
print(alexnet_ori._modules['features']._modules.keys())

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            'relu1': nn.ReLU(inplace=True),
            'maxpool1': nn.MaxPool2d(kernel_size=3, stride=2),
            'conv2': nn.Conv2d(64, 192, kernel_size=5, padding=2),
            'relu2': nn.ReLU(inplace=True),
            'maxpool2': nn.MaxPool2d(kernel_size=3, stride=2),
            'conv3': nn.Conv2d(192, 384, kernel_size=3, padding=1),
            'relu3': nn.ReLU(inplace=True),
            'conv4': nn.Conv2d(384, 256, kernel_size=3, padding=1),
            'relu4': nn.ReLU(inplace=True),
            'conv5': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu5': nn.ReLU(inplace=True),
            'maxpool5':nn.MaxPool2d(kernel_size=3, stride=2),
        }))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(OrderedDict({
            'dropout1': nn.Dropout(),
            'linear1': nn.Linear(256 * 6 * 6, 4096),
            'relu1': nn.ReLU(inplace=True),
            'dropout2': nn.Dropout(),
            'linear2': nn.Linear(4096, 4096),
            'relu2': nn.ReLU(inplace=True),
            'linear3': nn.Linear(4096, num_classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


alexnet = AlexNet(num_classes=2)
print(alexnet._modules['features']._modules.keys())


