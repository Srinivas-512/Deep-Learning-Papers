import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def apply_random_mask(img):
    y1, x1 = np.random.randint(0, 64, 2)
    y2, x2 = y1 + 64, x1 + 64
    masked_part = img[:, :, y1:y2, x1:x2]
    masked_img = img.clone()
    masked_img[:,:, y1:y2, x1:x2] = 0

    return masked_img, masked_part


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    random = torch.randn(0,1)
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace = True),

        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace = True),

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace = True),

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace = True),

        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace = True),

        nn.Conv2d(512, 4096, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(4096),
        nn.LeakyReLU(0.2, inplace = True)
    ) 
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(4096, 512, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace = True),

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace = True),

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace = True),

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),

        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(3),
        nn.Tanh()
    )
  
  def forward(self, x): # x is the incomplete image 
    x = self.encoder(x)
    x = self.decoder(x)
    return(x)
  
  def reconstruction_loss(self,x,real):
    x = self(x)
    loss_func = nn.MSELoss()
    loss = loss_func(real, x)
    l2 = (torch.mean(loss)).item()
    return l2
  
  def adv_loss(self, x):
    fake_images = self(x)
    fake_targets = torch.ones(1,4)
    fake_preds = obj_d(fake_images)
    adv_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    return adv_loss.item()


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.discriminator = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace = True),

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace = True),

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace = True),

        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
        nn.Flatten(),
        nn.Sigmoid()
    )
  
  def forward(self, x):
    x = self.discriminator(x)
    return x
  
  def loss(self, real_images, x):
    fake_images = obj_g(x)
    fake_preds = self(fake_images)
    fake_targets = torch.zeros(1,4)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    real_preds = self(real_images)
    real_targets = torch.ones(1,4)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    loss = fake_loss + real_loss
    return loss.item()

obj_g = Generator()
obj_d = Discriminator()