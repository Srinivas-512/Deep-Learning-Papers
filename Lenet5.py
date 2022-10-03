import torch
import torch.nn.module
import torch.nn.functional as func


class LeNet5(torch.nn.Module):
    def __init__(self, **kwargs):
        super(LeNet5, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(1, 6, kernel_size = 5, stride=1, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2, padding=0, stride=2)
        self.conv_layer2 = torch.nn.Conv2d(6, 16, kernel_size = 5, stride=1, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 2, padding=0, stride=2)
        self.conv_layer3 = torch.nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.classifier1 = torch.nn.Linear(in_features=120 , out_features=84)
        self.classifier2 = torch.nn.Linear(in_features=84 , out_features=10)

    def forward(self, x):
        out = self.pool1(func.relu(self.conv_layer1(x)))
        out = self.pool2(func.relu(self.conv_layer2(out)))
        out = func.relu(self.conv_layer3(out))
        out = out.reshape(out.size(0), -1)
        self.dropout = torch.nn.Dropout(0.5) 
        out = func.relu(self.classifier1(out))
        self.dropout = torch.nn.Dropout(0.5) 
        out = self.classifier2(out)
        out = func.log_softmax(out)
        return out