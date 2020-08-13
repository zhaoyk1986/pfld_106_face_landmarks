#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_bn1(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                  groups=in_channels, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()
        self.base_channel = 16
        self.model = nn.Sequential(
            conv_bn(3, self.base_channel, 2),  # 56*56
            conv_dw(self.base_channel, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2),
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
        )
        self.model1 = nn.Sequential(
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2),
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 8, 2),
            conv_dw(self.base_channel * 8, self.base_channel * 16, 1),
            conv_dw(self.base_channel * 16, self.base_channel * 16, 1))

        self.extra = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 16, out_channels=self.base_channel * 4, kernel_size=1),
            nn.ReLU(),
            SeperableConv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 8,
                            kernel_size=3, stride=2, padding=1),
        )
        conv_dw(self.base_channel * 4, self.base_channel * 8, 2),
        conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
        conv_dw(self.base_channel * 8, self.base_channel * 8, 2),
        conv_dw(self.base_channel * 8, self.base_channel * 16, 1),
        conv_dw(self.base_channel * 16, self.base_channel * 16, 1)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 106 * 2)

    def forward(self, x):  # x: 3, 112, 112
        out = self.model(x)
        x = self.model1(out)
        x = self.extra(x)
        x = self.avg_pool1(x)
        x = x.view(x.size(0), -1)
        landmarks = self.fc(x)
        return out, landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn1(64, 128, 3, 2)
        self.conv2 = conv_bn1(128, 128, 3, 1)
        self.conv3 = conv_bn1(128, 32, 3, 2)
        self.conv4 = conv_bn1(32, 128, 3, 1)
        self.max_pool1 = nn.AdaptiveMaxPool2d(1)
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
    dummy_input = torch.randn(1, 3, 224, 224)
    plfd_backbone = PFLDInference()
    # print(plfd_backbone)
    # torch.save(plfd_backbone.state_dict(), 'lite.pth')
    from thop import profile

    macs, p = profile(model=plfd_backbone, inputs=(dummy_input, ), verbose=False)
    print(f"macs: {macs / 1000000.0}, params: {p / 1000000.0}")
    auxiliarynet = AuxiliaryNet()
    import time
    tic = time.time()
    N = 100
    for i in range(N):
        features, landmarks_ = plfd_backbone(dummy_input)
    average_infer_time = (time.time() - tic) / N
    print("averager inference time: {:.4f}, FPS: {:.2f}".format(average_infer_time / dummy_input.shape[0],
                                                                1 * dummy_input.shape[0] / average_infer_time))
    angle = auxiliarynet(features)
    print("angle.shape:{0:}, landmarks.shape: {1:}".format(
        angle.shape, landmarks_.shape))
