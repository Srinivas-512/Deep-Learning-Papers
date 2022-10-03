import torch
import torch.nn.module
import torch.nn.functional as func

class Alexnet(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Alexnet, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(3, 96, kernel_size = 11, stride=4, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 3, padding=0, stride=2)
        self.conv_layer2 = torch.nn.Conv2d(96, 256, kernel_size = 5, stride=1, padding=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 3, padding=0, stride=2)
        self.conv_layer3 = torch.nn.Conv2d(256, 384, kernel_size = 3, stride=1, padding=1)
        self.conv_layer4 = torch.nn.Conv2d(384, 384, kernel_size = 3, stride=1, padding=1)
        self.conv_layer5 = torch.nn.Conv2d(384, 256, kernel_size = 3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size = 3, padding=0, stride=2)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier1 = torch.nn.Linear(in_features= 9216, out_features= 4096)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier2 = torch.nn.Linear(in_features= 4096, out_features= 4096)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier3 = torch.nn.Linear(in_features=4096 , out_features=10) 

    def forward(self, x):
        out = self.pool1(func.relu(self.conv_layer1(x)))
        out = self.pool2(func.relu(self.conv_layer2(out)))
        out = func.relu(self.conv_layer3(out))
        out = func.relu(self.conv_layer4(out))
        out = self.pool3(func.relu(self.conv_layer5(out)))
        out = out.reshape(out.size(0), -1)
        #out = self.dropout(out)
        out = func.relu(self.classifier1(out))
        #out = self.dropout(out)
        out = func.relu(self.classifier2(out))
        out = func.log_softmax(self.classifier3(out))
        # check dropout 
        return out