import torch
import torch.nn.module
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv_layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv_layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv_layer3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv_layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv_layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv_layer3(x)
        x = self.batch_norm3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, block):
        super(ResNet50, self).__init__()
        self.expansion = 4
        layers = [3, 4, 6, 3]
        self.in_channels = 64
        self.conv_layer1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.res_layer1 = self.construct_layers(block, layers[0], 64, 1)
        self.res_layer2 = self.construct_layers(block, layers[1], 128, 2)
        self.res_layer3 = self.construct_layers(block, layers[2], 256, 2)
        self.res_layer4 = self.construct_layers(block, layers[3], 512, 2)
        self.avgpool_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier1 = nn.Linear(512 * self.expansion, 10)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool_layer(x)

        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)

        x = self.avgpool_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier1(x)
        return x

    def construct_layers(self, block, num_residual_blocks, intermediate_channels, stride):
        res_layers = []
        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        res_layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion
        for i in range(num_residual_blocks - 1):
            res_layers.append(block(self.in_channels, intermediate_channels))
        return nn.Sequential(*res_layers)