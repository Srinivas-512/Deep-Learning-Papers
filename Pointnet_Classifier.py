import torch
import torch.nn.functional as func
import torch.nn as nn

# Multi-Layer perceptrons
class MLP_two_layers(nn.Module):
  def __init__(self, in_channels, layer1_channels, layer2_channels):
    super(MLP_two_layers, self).__init__()
    self.layer1 = torch.nn.Linear(in_channels, layer1_channels)
    self.layer2 = torch.nn.Linear(layer1_channels, layer2_channels)

  def forward(self, x):
    x = self.layer1(x)
    x = func.relu(x)
    x = self.layer2(x)
    x = func.relu(x)
    return x 


class MLP_three_layers(nn.Module):
  def __init__(self, in_channels, layer1_channels, layer2_channels, layer3_channels):
    super(MLP_three_layers, self).__init__()
    self.layer1 = torch.nn.Linear(in_channels, layer1_channels)
    self.layer2 = torch.nn.Linear(layer1_channels, layer2_channels)
    self.layer3 = torch.nn.Linear(layer2_channels, layer3_channels)

  def forward(self, x):
    x = self.layer1(x)
    x = func.relu(x)
    x = self.layer2(x)
    x = func.relu(x)
    x = self.layer3(x)
    x = func.relu(x)
    return x 


# T-Net block
class Transform(nn.Module):
  def __init__(self, in_channels, reshape_value):
    super(Transform, self).__init__()
    self.reshape_value = reshape_value
    self.layer1 = torch.nn.Linear(in_channels, 64)
    self.layer2 = torch.nn.Linear(64, 128)
    self.layer3 = torch.nn.Linear(128, 1024)
    #self.pool = torch.nn.MaxPool1d(kernel_size=3 , padding=1 , stride=1)
    self.pool = torch.nn.MaxPool1d(2048,2048)
    self.fc1 = torch.nn.Linear(1024, 512)
    self.fc2 = torch.nn.Linear(512, 256)
    self.trainable_weights = torch.rand((256,(reshape_value*reshape_value)), requires_grad=True)
    self.trainable_biases = torch.rand((1,(reshape_value*reshape_value)), requires_grad=True)
  
  def forward(self,x):
    x = func.relu(self.layer1(x))
    x = func.relu(self.layer2(x))
    x = func.relu(self.layer3(x))
    x = self.pool(x.permute(0,2,1)).permute(0,2,1)
    x = func.relu(self.fc1(x))
    x = func.relu(self.fc2(x))
    x = x@self.trainable_weights
    x = x + self.trainable_biases
    x = x.reshape(4,self.reshape_value,self.reshape_value)
    return x

# Classifier
class Classification(nn.Module):
  def __init__(self, points):
    super(Classification, self).__init__()
    self.points = points
    self.spatial_transformer = Transform(3, 3)
    self.feature_transformer = Transform(64, 64)
    self.two_layer_mlp = MLP_two_layers(3,64,64)
    self.three_layer_mlp_first = MLP_three_layers(64,64,128,1024)
    k=10
    self.three_layer_mlp_second = MLP_three_layers(1024,512,256,k)

  def forward(self):
    self.points = self.points@self.spatial_transformer(self.points)
    self.points = self.two_layer_mlp(self.points)
    self.points = self.points@self.feature_transformer(self.points)
    self.points = self.three_layer_mlp_first(self.points)
    self.points = torch.nn.MaxPool1d(2048,2048)(self.points.permute(0,2,1)).permute(0,2,1)
    self.points = self.three_layer_mlp_second(self.points)
    return (self.points).shape
