#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/8/11 12:10 下午
# @Author : Xintao
# @File : ghost_pfld.py

# -*- coding: utf-8 -*-
# @Time : 2020/4/10 9:50 下午
# @Author : Xintao
# @File : mobilev3_pfld.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_bn(inp, oup, kernel_size, stride, padding=1, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, kernel_size, stride, padding, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # F.avg_pool2d()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print((x * y == x * y.expand_as(x)).all().all())
        return x * y


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se=False):
        super(GhostBottleneck, self).__init__()
        has_se = se
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SEModule(mid_chs)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()

        self.conv_bn1 = conv_bn(3, 16, 3, stride=1)
        self.conv_bn2 = GhostBottleneck(16, 64, 16, 3, 2, se=False)

        self.conv3_1 = GhostBottleneck(16, 64, 24, 3, 2, se=False)

        self.block3_2 = GhostBottleneck(24, 72, 24, 3, 1, se=False)
        self.block3_3 = GhostBottleneck(24, 72, 40, 5, 1, se=True)
        self.block3_4 = GhostBottleneck(40, 120, 40, 5, 1, se=True)
        self.block3_5 = GhostBottleneck(40, 120, 40, 5, 1, se=True)

        self.conv4_1 = GhostBottleneck(40, 240, 80, 3, 2, se=False)

        self.conv5_1 = GhostBottleneck(80, 200, 80, 3, 1, se=False)
        self.block5_2 = GhostBottleneck(80, 480, 112, 3, 1, se=True)
        self.block5_3 = GhostBottleneck(112, 672, 112, 3, 1, se=True)
        self.block5_4 = GhostBottleneck(112, 672, 160, 3, 1, se=True)
        # self.block5_5 = MobileBottleneck(160, 160, 3, 1, 960, True, "HS")

        self.conv6_1 = GhostBottleneck(160, 320, 16, 3, 1, se=False)  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        # self.conv8 = conv_bn(32, 128, 7, 1, padding=0, nlin_layer=Hswish)  # [128, 1, 1]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)
        self.hs = Hswish()
        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 106 * 2)

    def forward(self, x):  # x: 3, 112, 112
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)

        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        # x = self.block5_5(x)
        # x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.hs(self.conv8(x))
        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return out1, landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(40, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 3, 1, padding=0)
        self.max_pool1 = nn.MaxPool2d(5)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 112, 112)
    plfd_backbone = PFLDInference()
    print(plfd_backbone)
    from thop import profile

    macs, param = profile(model=plfd_backbone, inputs=(dummy_input, ), verbose=False)
    print(f"macs: {macs / 1000000}, params: {param / 1000000}")
    auxiliarynet = AuxiliaryNet()
    import time
    tic = time.time()
    N = 10
    for i in range(N):
        features, landmarks_ = plfd_backbone(dummy_input)
    average_infer_time = (time.time() - tic) / N
    print("averager inference time: {:.4f}, FPS: {:.2f}".format(average_infer_time / dummy_input.shape[0],
                                                                1 * dummy_input.shape[0] / average_infer_time))
    angle = auxiliarynet(features)
    print("angle.shape:{0:}, landmarks.shape: {1:}".format(
        angle.shape, landmarks_.shape))
