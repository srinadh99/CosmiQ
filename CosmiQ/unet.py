# unet_batchnorm.py
from torch import sigmoid
import torch.nn as nn
from parts import *


class UNet2Sigmoid(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=32):
        super(type(self), self).__init__()
        self.inc = inconv(n_channels, hidden)
        self.down1 = down(hidden, hidden * 2)
        self.up8 = up(hidden * 2, hidden)
        self.outc = outconv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up8(x2, x1)
        x = self.outc(x)
        return sigmoid(x)


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=32):
        super(type(self), self).__init__()
        self.inc = inconv(n_channels, hidden)
        self.down1 = down(hidden, hidden * 2)
        self.up8 = up(hidden * 2, hidden)
        self.outc = outconv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up8(x2, x1)
        x = self.outc(x)
        return x


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=32):
        super(type(self), self).__init__()
        self.inc = inconv(n_channels, hidden)
        self.down1 = down(hidden, hidden * 2)
        self.down2 = down(hidden * 2, hidden * 4)
        self.up7 = up(hidden * 4, hidden * 2)
        self.up8 = up(hidden * 2, hidden)
        self.outc = outconv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up7(x3, x2)
        x = self.up8(x, x1)
        x = self.outc(x)
        return x


class UNet2SigmoidVI(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=32, sigma_prior=1.0):
        super(type(self), self).__init__()
        self.inc   = inconv_vi(n_channels, hidden, sigma_prior=sigma_prior)
        self.down1 = down_vi(hidden, hidden * 2, sigma_prior=sigma_prior)
        self.up8   = up_vi(hidden * 2, hidden, sigma_prior=sigma_prior)
        self.outc  = outconv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # [B, hidden, H, W]
        x2 = self.down1(x1)   # [B, 2*hidden, H/2, W/2]
        x  = self.up8(x2, x1) # [B, hidden, H, W]
        x  = self.outc(x)     # [B, n_classes, H, W]
        return sigmoid(x)

    def kl_loss(self):
        kl = 0.0
        for m in self.modules():
            if isinstance(m, BayesianConv2d):
                kl = kl + m.kl_divergence()
        return kl
    
class WrappedModel(nn.Module):
    def __init__(self, network):
        super(type(self), self).__init__()
        self.module = network

    def forward(self, *x):
        return self.module(*x)
