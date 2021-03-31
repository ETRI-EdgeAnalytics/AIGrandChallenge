import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, kernal_size, stride, bias=False):
        super(DepthwiseConv2D, self).__init__()
        padding = (kernal_size - 1) // 2

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernal_size, padding=padding, stride=stride, groups=in_channels, bias=bias)

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out


class GroupConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, n_chunks=1, bias=False):
        super(GroupConv2D, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = split_layer(in_channels, n_chunks)
        split_out_channels = split_layer(out_channels, n_chunks)

        if n_chunks == 1:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        else:
            self.group_layers = nn.ModuleList()
            for idx in range(n_chunks):
                self.group_layers.append(nn.Conv2d(self.split_in_channels[idx], split_out_channels[idx], kernel_size=kernel_size, bias=bias))

    def forward(self, x):
        if self.n_chunks == 1:
            return self.group_conv(x)
        else:
            split = torch.split(x, self.split_in_channels, dim=1)
            out = torch.cat([layer(s) for layer, s in zip(self.group_layers, split)], dim=1)
            return out

class MDConv(nn.Module):
    def __init__(self, out_channels, n_chunks, stride=1, bias=False):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_out_channels = split_layer(out_channels, n_chunks)

        self.layers = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            self.layers.append(DepthwiseConv2D(self.split_out_channels[idx], kernal_size=kernel_size, stride=stride, bias=bias))

    def forward(self, x):
        split = torch.split(x, self.split_out_channels, dim=1)
        out = torch.cat([layer(s) for layer, s in zip(self.layers, split)], dim=1)
        return out


def round_filters(filters, multiplier=1.0, divisor=8, min_depth=None):
    multiplier = multiplier
    divisor = divisor
    min_depth = min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, swish):
        super(SqueezeExcitation, self).__init__()
        self.activation = Swish() if swish else nn.ReLU()

        self.se_reduce = nn.Sequential(
            GroupConv2D(in_channels, out_channels, bias=True),
            nn.BatchNorm2d(out_channels),
            self.activation
        )

        self.se_expand = nn.Sequential(
            GroupConv2D(out_channels, in_channels, bias=True),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        se_tensor = torch.mean(x, dim=[2, 3], keepdim=True)
        out = self.se_expand(self.se_reduce(se_tensor))
        out = torch.sigmoid(out) * x

        return out

class MixBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_chunks, stride, expand_ratio, se_ratio, swish, expand_ksize, project_ksize):
        super(MixBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
        self.activation = Swish() if swish else nn.ReLU()

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                GroupConv2D(in_channels, in_channels * expand_ratio, n_chunks=expand_ksize),
                nn.BatchNorm2d(in_channels * expand_ratio),
                self.activation
            )

            self.mdconv = nn.Sequential(
                MDConv(in_channels * expand_ratio, n_chunks=n_chunks, stride=stride),
                nn.BatchNorm2d(in_channels * expand_ratio),
                self.activation
            )

            if self._has_se:
                num_reduced_filters = max(1, int(in_channels * se_ratio))
                self.squeeze_excitation = SqueezeExcitation(in_channels * expand_ratio, num_reduced_filters, swish)
            self.project_conv = nn.Sequential(
                GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.mdconv = nn.Sequential(
                MDConv(in_channels, n_chunks=n_chunks, stride=stride),
                nn.BatchNorm2d(in_channels),
                self.activation
            )

            if self._has_se:
                num_reduced_filters = max(1, int(in_channels * se_ratio))
                self.squeeze_excitation = SqueezeExcitation(in_channels, num_reduced_filters, swish)
            self.project_conv = nn.Sequential(
                GroupConv2D(in_channels, out_channels, n_chunks=project_ksize),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.expand_ratio != 1:
            out = self.expand_conv(x)
            out = self.mdconv(out)

            if self._has_se:
                out = self.squeeze_excitation(out)
            out = self.project_conv(out)
        else:
            out = self.mdconv(x)
            if self._has_se:
                out = self.squeeze_excitation(out)
            out = self.project_conv(out)

        if self.stride == 1 and self.in_channels == self.out_channels:
            out = out + x
        return out

class MixNet(nn.Module):
    def __init__(self, stem, head, last_out_channels, block_args, dropout_rate=0.2, num_classes=1000, dataset='IMAGENET'):
        super(MixNet, self).__init__()
        self.dataset = dataset

        if dataset == 'IMAGENET':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=stem, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(stem),
                nn.ReLU(),
            )
        elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=stem, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(stem),
                nn.ReLU(),
            )

        layers = []
        for in_channels, out_channels, n_chunks, stride, expqand_ratio, se_ratio, swish, expand_ksize, project_ksize in block_args:
            layers.append(MixBlock(in_channels, out_channels, n_chunks, stride, expqand_ratio, se_ratio, swish, expand_ksize, project_ksize))

        self.layers = nn.Sequential(*layers)

        self.head_conv = nn.Sequential(
            nn.Conv2d(last_out_channels, head, kernel_size=1),
            nn.BatchNorm2d(head),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(head, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        out = self.head_conv(out)

        if self.dataset == 'IMAGENET':
            out = F.avg_pool2d(out, 7)
        elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def mixnet_s(num_classes=100, multiplier=1.0, divisor=8, min_depth=None, dataset='CIFAR100'):
    if dataset == 'IMAGENET':
        small = [
            [16, 16, 1, 1, 1, None, False, 1, 1],
            [16, 24, 1, 2, 6, None, False, 2, 2],
            [24, 24, 1, 1, 3, None, False, 2, 2],
            [24, 40, 3, 2, 6, 0.5, True, 1, 1],
            [40, 40, 2, 1, 6, 0.5, True, 2, 2],

            [40, 40, 2, 1, 6, 0.5, True, 2, 2],
            [40, 40, 2, 1, 6, 0.5, True, 2, 2],
            [40, 80, 3, 2, 6, 0.25, True, 1, 2],
            [80, 80, 2, 1, 6, 0.25, True, 1, 2],
            [80, 80, 2, 1, 6, 0.25, True, 1, 2],

            [80, 120, 3, 1, 6, 0.5, True, 2, 2],
            [120, 120, 4, 1, 3, 0.5, True, 2, 2],
            [120, 120, 4, 2, 3, 0.5, True, 2, 2],
            [120, 200, 5, 1, 6, 0.5, True, 1, 1],
            [200, 200, 4, 1, 6, 0.5, True, 1, 2],

            [200, 200, 4, 1, 6, 0.5, True, 1, 2]
        ]
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        small = [
            [16, 16, 1, 1, 1, None, False, 1, 1],
            [16, 24, 1, 1, 6, None, False, 2, 2],
            [24, 24, 1, 1, 3, None, False, 2, 2],
            [24, 40, 3, 2, 6, 0.5, True, 1, 1],
            [40, 40, 2, 1, 6, 0.5, True, 2, 2],

            [40, 40, 2, 1, 6, 0.5, True, 2, 2],
            [40, 40, 2, 1, 6, 0.5, True, 2, 2],
            [40, 80, 3, 2, 6, 0.25, True, 1, 2],
            [80, 80, 2, 1, 6, 0.25, True, 1, 2],
            [80, 80, 2, 1, 6, 0.25, True, 1, 2],

            [80, 120, 3, 1, 6, 0.5, True, 2, 2],
            [120, 120, 4, 1, 3, 0.5, True, 2, 2],
            [120, 120, 4, 2, 3, 0.5, True, 2, 2],
            [120, 200, 5, 1, 6, 0.5, True, 1, 1],
            [200, 200, 4, 1, 6, 0.5, True, 1, 2],

            [200, 200, 4, 1, 6, 0.5, True, 1, 2]
        ]
    stem = round_filters(16, multiplier)
    last_out_channels = round_filters(200, multiplier)
    head = round_filters(1536, multiplier)

    return MixNet(stem=stem, head=head, last_out_channels=last_out_channels, block_args=small, num_classes=num_classes, dataset=dataset)


def mixnet_m(num_classes=100, multiplier=1.0, divisor=8, min_depth=None, dataset='CIFAR100'):
    if dataset == 'IMAGENET':
        medium = [
            [24, 24, 1, 1, 1, None, False, 1, 1],
            [24, 32, 3, 2, 6, None, False, 2, 2],
            [32, 32, 1, 1, 3, None, False, 2, 2],
            [32, 40, 4, 2, 6, 0.5, True, 1, 1],
            [40, 40, 2, 1, 6, 0.5, True, 2, 2],

            [40, 40, 2, 1, 6, 0.5, True, 2, 2],
            [40, 40, 2, 1, 6, 0.5, True, 2, 2],
            [40, 80, 3, 2, 6, 0.25, True, 1, 1],
            [80, 80, 4, 1, 6, 0.25, True, 2, 2],
            [80, 80, 4, 1, 6, 0.25, True, 2, 2],

            [80, 80, 4, 1, 6, 0.25, True, 2, 2],
            [80, 120, 1, 1, 6, 0.5, True, 1, 1],
            [120, 120, 4, 1, 3, 0.5, True, 2, 2],
            [120, 120, 4, 1, 3, 0.5, True, 2, 2],
            [120, 120, 4, 1, 3, 0.5, True, 2, 2],

            [120, 200, 4, 2, 6, 0.5, True, 1, 1],
            [200, 200, 4, 1, 6, 0.5, True, 1, 2],
            [200, 200, 4, 1, 6, 0.5, True, 1, 2],
            [200, 200, 4, 1, 6, 0.5, True, 1, 2]
        ]
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        medium = [
            [24, 24, 1, 1, 1, None, False, 1, 1],
            [24, 32, 3, 1, 6, None, False, 2, 2],
            [32, 32, 1, 1, 3, None, False, 2, 2],
            [32, 40, 4, 2, 6, 0.5, True, 1, 1],
            [40, 40, 2, 1, 6, 0.5, True, 2, 2],

            [40, 40, 2, 1, 6, 0.5, True, 2, 2],
            [40, 40, 2, 1, 6, 0.5, True, 2, 2],
            [40, 80, 3, 2, 6, 0.25, True, 1, 1],
            [80, 80, 4, 1, 6, 0.25, True, 2, 2],
            [80, 80, 4, 1, 6, 0.25, True, 2, 2],

            [80, 80, 4, 1, 6, 0.25, True, 2, 2],
            [80, 120, 1, 1, 6, 0.5, True, 1, 1],
            [120, 120, 4, 1, 3, 0.5, True, 2, 2],
            [120, 120, 4, 1, 3, 0.5, True, 2, 2],
            [120, 120, 4, 1, 3, 0.5, True, 2, 2],

            [120, 200, 4, 2, 6, 0.5, True, 1, 1],
            [200, 200, 4, 1, 6, 0.5, True, 1, 2],
            [200, 200, 4, 1, 6, 0.5, True, 1, 2],
            [200, 200, 4, 1, 6, 0.5, True, 1, 2]
        ]
    for line in medium:
        line[0] = round_filters(line[0], multiplier)
        line[1] = round_filters(line[1], multiplier)

    stem = round_filters(24, multiplier)
    last_out_channels = round_filters(200, multiplier)
    head = round_filters(1536, multiplier=1.0)

    return MixNet(stem=stem, head=head, last_out_channels=last_out_channels, block_args=medium, dropout_rate=0.25, num_classes=num_classes, dataset=dataset)


def mixnet_l(num_classes=100, dataset='CIFAR100'):
    return mixnet_m(num_classes=num_classes, multiplier=1.3, dataset=dataset)


def mixnet(conf)
    return mixnet_s()

#model = mixnet_s()
#print(model)
#from thop import profile
#input = torch.randn(1,3,32,32)
#mac,params = profile(model, inputs=(input,))
#print('mac:',mac,'params:',params)
