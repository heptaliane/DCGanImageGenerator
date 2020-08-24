# -*- coding: utf-8 -*-
import torch.nn as nn


class DCGanDiscriminatorConv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyRelu(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        return self.relu(x)


class DCGanDiscriminator(nn.Module):
    def __init__(self, in_ch, depth=4, detach=-1):
        super().__init__()

        feature = 1024 // 2 ** (depth - 1)
        self.detach = detach

        self.layers = list()
        self.layers.append(DCGanDiscriminatorConv(in_ch, feature, False))
        for _ in range(1, depth):
            feature *= 2
            self.layers.append(DCGanDiscriminatorConv(feature // 2, feature))

        self.pred = nn.Sequential(
            nn.Conv2d(feature, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == self.detach:
                x.detach()
            x = layer(x)
        return self.pred(x)
