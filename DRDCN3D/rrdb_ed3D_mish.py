import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.BatchNorm3d(in_features)]
            layers += [mish()]
            layers += [nn.Conv3d(in_features, filters, 3, 1, 1, bias=True)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x
        # return out + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)#, DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
        # return self.dense_blocks(x) + x


class RRDB(nn.Module):
    def __init__(self, in_channels, out_channels, filters=48, num_res_blocks=1):
        super(RRDB, self).__init__()
        self.out_channels = out_channels

        # encoder
        # First layer
        self.conv1 = nn.Conv3d(in_channels, filters, kernel_size=3, stride=2, padding=1)
        self.res_blocks_e = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks+0)])

        self.transdown = nn.Sequential(
            nn.BatchNorm3d(filters),
            mish(),
            nn.Conv3d(filters, filters, kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        self.res_blocks_c = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks+1)])

        # decoder
        self.transup1 = nn.Sequential(
            nn.BatchNorm3d(filters),
            mish(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding=1),
        )

        self.res_blocks_d = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks+0)])
        self.transup2 = nn.Sequential(
            nn.BatchNorm3d(filters),
            mish(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(filters, out_channels, kernel_size=3, stride=1, padding=(0,1,1)),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.res_blocks_e(out1)
        out3 = self.transdown(out2)

        out4 = self.res_blocks_c(out3)
        out = torch.add(out3, out4)
        out5 = self.transup1(out)
        out6 = self.res_blocks_d(out5)
        out = self.transup2(out6)

        # softplus activation for the concentrations
        out[:,:self.out_channels-1] = F.softplus(out[:,:self.out_channels-1].clone(), beta=5)
        # sigmoid activation for the hydraulic heads
        out[:,self.out_channels-1] = torch.sigmoid(out[:,self.out_channels-1])

        return out

    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))


if __name__ == '__main__':
    model = RRDB(1, 6)
    print("number of parameters: {}\nnumber of layers: {}".format(*model._num_parameters_convlayers()))
