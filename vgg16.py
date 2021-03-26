import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from pruned_layers import *

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            PrunedConv(torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PrunedConv(torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PrunedConv(torch.nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PrunedConv(torch.nn.Conv2d(128, 128, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PrunedConv(torch.nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PrunedConv(torch.nn.Conv2d(256, 256, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PrunedConv(torch.nn.Conv2d(256, 256, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PrunedConv(torch.nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PrunedConv(torch.nn.Conv2d(512, 512, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PrunedConv(torch.nn.Conv2d(512, 512, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PrunedConv(torch.nn.Conv2d(512, 512, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PrunedConv(torch.nn.Conv2d(512, 512, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PrunedConv(torch.nn.Conv2d(512, 512, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifer = nn.Sequential(
            PrunedLinear(torch.nn.Linear(512, 512)),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            PrunedLinear(torch.nn.Linear(512, 512)),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            PrunedLinear(torch.nn.Linear(512, 10))
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.mean(3).mean(2)
        out = self.classifer(feat)
        return out

class VGG16_half(nn.Module):
    def __init__(self):
        super(VGG16_half, self).__init__()
        self.features = nn.Sequential(
            PrunedConv(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            PrunedConv(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PrunedConv(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PrunedConv(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PrunedConv(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PrunedConv(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PrunedConv(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PrunedConv(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PrunedConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PrunedConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PrunedConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PrunedConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PrunedConv(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifer = nn.Sequential(
            PrunedLinear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            PrunedLinear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            PrunedLinear(256, 10)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.mean(3).mean(2)
        out = self.classifer(feat)
        return out


class VGG16_5(nn.Module):
    def __init__(self):
        super(VGG16_5, self).__init__()
        self.features = nn.Sequential(
            PrunedConv(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PrunedConv(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            PrunedConv(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PrunedConv(128, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            PrunedConv(128, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PrunedConv(256, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PrunedConv(256, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            PrunedConv(256, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PrunedConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PrunedConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            PrunedConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PrunedConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            PrunedConv(512, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            )
        self.classifer = nn.Sequential(
            PrunedLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            PrunedLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            PrunedLinear(512, 10)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.mean(3).mean(2)
        out = self.classifer(feat)
        return out
