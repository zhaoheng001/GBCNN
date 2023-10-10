from numpy import floor, ceil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision
from convrf.convrf import Conv2dRF
class SparseNet(nn.Module):
    def __init__(self):
        super(SparseNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding = 'same'),
            #nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding = 'same'),
            #nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 48, 1, padding = 'same'),
            #nn.BatchNorm2d(48, affine=False),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 32, 5, padding = 'same'),
            #nn.BatchNorm2d(32, affine=False),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, 5, padding = 'same'),
            #nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 1, 5, padding = 'same'),
            #nn.BatchNorm2d(3, affine=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

class SDPF(nn.Module):
    def __init__(self):
        super(SDPF, self).__init__()
        self.layer1 = nn.Sequential(
            Conv2dRF(1, 64, 5, padding = 2, fbank_type="pdefine", kernel_drop_rate=0.93),
            #nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
        )
        # self.layer2 = nn.Sequential(
        #     Conv2dRF(64, 64, 5, padding = 2, fbank_type="pdefine", kernel_drop_rate=0.93),
        #     #nn.BatchNorm2d(64, affine=False),
        #     nn.ReLU(inplace=True),
        # )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding = 'same'),
            #nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 48, 1, padding = 'same'),
            #nn.BatchNorm2d(48, affine=False),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 32, 5, padding = 'same'),
            #nn.BatchNorm2d(32, affine=False),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, 5, padding = 'same'),
            #nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 1, 5, padding = 'same'),
            #nn.BatchNorm2d(3, affine=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
