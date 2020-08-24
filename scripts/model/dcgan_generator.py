# -*- coding: utf-8 -*-
import torch.nn as nn


class DCGanGeneratorConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding,
                                       bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class DCGanGenerator(nn.Module):
    def __init__(self, in_ch, out_ch=3, depth=4, detach=-1):
        super().__init__()

        self.detach = detach
        feature = 1024

        layers = list()
        layers.append(DCGanGeneratorConv(in_ch, feature, 4, 1, 0))

        for _ in range(1, depth):
            feature = feature // 2
            layers.append(DCGanGeneratorConv(feature * 2, feature, 4, 2, 1))

        self.layers = nn.Sequential(*layers)
        self.pred = nn.Sequential(
            nn.ConvTranspose2d(feature, out_ch, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == self.detach:
                x.detach()
            x = layer(x)
        return self.pred(x)
