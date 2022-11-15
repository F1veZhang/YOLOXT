import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .deform_conv_v2 import *


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
        
class SoftPool2D(nn.Module):
    def __init__(self):
        super(SoftPool2D, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32,modulation=False):
        super(CoordAtt, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        
        self.conv0 = nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.bn0 = nn.BatchNorm2d(128)
        self.act = h_swish()

        # self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        if modulation:
            self.conv_h = DeformConv2d(mip, oup, 3, padding=1, bias=False, modulation=modulation)
            self.conv_w = DeformConv2d(mip, oup, 3, padding=1, bias=False, modulation=modulation)
        else:
            self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
            self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n,c,h,w = x.size()
        # x_h = self.pool_h(x)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x = self.conv0(x)
        x = self.bn0(x)

        aa = nn.AdaptiveAvgPool2d((None, 1))
        x_h = aa(x)
        bb = nn.AdaptiveAvgPool2d((1, None))
        x_w = bb(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


