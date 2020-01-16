# -*- coding: utf-8 -*-
"""Implements convolutional adversarial auto-encoder

TODO:

"""

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    eps = torch.randn_like(std)

    return mu + eps*std


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.BatchNorm2d(in_features)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)#, DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class Encoder(nn.Module):
    def __init__(self, inchannels=1, outchannels=1, filters=48, num_res_blocks=1):
        super(Encoder, self).__init__()
        # input size, inchannels x 40 x 80
        self.conv1 = nn.Conv2d(inchannels, filters, kernel_size=3, stride=2, padding=1)
        # state size. filters x 20 x 40
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # state size. filters x 20 x 40
        self.trans = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1),
        )

        self.mu = nn.Conv2d(filters, outchannels, 3, 1, 1, bias=False)
        self.logvar = nn.Conv2d(filters, outchannels, 3, 1, 1, bias=False)

    def forward(self, img):
        # img: inchannels x 40 x 80
        out1 = self.conv1(img)        # filters x 20 x 40
        out2 = self.res_blocks(out1)   # filters x 20 x 40
        out3 = self.trans(out2)        # filters x 10 x 20

        mu, logvar = self.mu(out3), self.logvar(out3)
        z = reparameterization(mu, logvar)  # latent dimension = outchannels*10*20
        return z

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

class Decoder(nn.Module):
    def __init__(self, inchannels=1, outchannels=1, filters=48, num_res_blocks=1):
        super(Decoder, self).__init__()

        # First layer. input size, inchannels x 10 x 20
        self.conv1 = nn.Conv2d(inchannels, filters, kernel_size=3, stride=1, padding=1)

        # state size. filters x 20 x 40
        # Residual blocks
        self.res_block1 = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks+1)])
        self.transup1 = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
        )
        self.res_block2 = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.transup2 = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(filters, outchannels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        # z: in_channels x 10 x 20
        out1 = self.conv1(z)          # filters x 10 x 20
        out2 = self.res_block1(out1)   # filters x 10 x 20
        out = torch.add(out1, out2)   # filters x 10 x 20
        out3 = self.transup1(out)      # filters x 20 x 40
        out4 = self.res_block2(out3)   # filters x 20 x 40
        img = self.transup2(out4)      # outchannels x 40 x 80

        return img

    def _n_parameters(self):
        n_params= 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params


class Discriminator(nn.Module):
    def __init__(self, inchannels=1, filters=48):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (inchannels) x 10 x 20
            nn.Conv2d(inchannels, filters, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (filters) x 5 x 10
            nn.Conv2d(filters, filters, 3, 1, 1, bias=True),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (filters) x 3 x 5
        )

        self.fc1 = nn.Sequential(
            nn.Linear(filters * 5 * 10,128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, inchannels),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        output1 = self.fc1(output)
        output2 = self.fc2(output1)
        return output2

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params
