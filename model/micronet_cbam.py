'''MicroNet'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
import math
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function

import sys
sys.path.append('../')
from regularization import *

def to_var(x, device, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    return Variable(x, requires_grad=requires_grad, volatile=volatile).to(device)

class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.relu = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu(x + 3.0) / 6.0

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(x)
        return out

@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        #return Func.mish(input)
        return mish(input)

class ConvSigmoid(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvSigmoid, self).__init__()
        self.cs = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size),
                                Sigmoid() )
    def forward(self, x):
        x = self.cs(x)
        return x


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, momentum=0.01, dilation=1, groups=1,
                       device = 'cuda:0' if cuda.is_available() else 'cpu',norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvReLU, self).__init__()
        self.relu6 = relu6
        self.cr = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
                                 nn.ReLU6(True) if relu6 else nn.ReLU(False)
                                )

    def forward(self, x):
        x = self.cr(x)
        return x

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, momentum=0.01, dilation=1, groups=1,
                       device = 'cuda:0' if cuda.is_available() else 'cpu', norm_layer=nn.BatchNorm2d,  **kwargs):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
        )


class ConvBNHSwish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, momentum=0.01,
                 dilation=1, groups=1, device = 'cuda:0' if cuda.is_available() else 'cpu',norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvBNHSwish, self).__init__()
        self.cb = ConvBN(in_channels, out_channels, kernel_size, stride, padding, bias, momentum, dilation, groups, device, norm_layer)
        self.act = HSwish(True)

    def forward(self, x):
        x = self.act(self.cb(x))
        return x

class ConvBNMish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, momentum=0.01,
                 dilation=1, groups=1, device = 'cuda:0' if cuda.is_available() else 'cpu',norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvBNMish, self).__init__()
        self.cb = ConvBN(in_channels, out_channels, kernel_size, stride, padding, bias, momentum, dilation, groups, device, norm_layer)
        self.act = Mish()

    def forward(self, x):
        x = self.act(self.cb(x))
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, momentum=0.01, \
                 dilation=1, groups=1, device = 'cuda:0' if cuda.is_available() else 'cpu', norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvBNReLU, self).__init__(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        nn.ReLU(False),
        )


class SEModule(nn.Module):
    def __init__(self, out_planes, number, kernel_size=1, bias=True, device = 'cuda:0' if cuda.is_available() else 'cpu', norm_layer = nn.BatchNorm2d):
        super(SEModule, self).__init__()
        self.bn = norm_layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_planes, number, kernel_size=1, bias=bias),
            nn.Conv2d(number, out_planes, kernel_size=1, bias=bias),
            Sigmoid()
        )

    def forward(self, x):
        out  = self.fc(self.avg_pool(x))
        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MicroBlock(nn.Module):
    '''expand + depthwise + pointwise
    Activation : ReLU or HSwish
    
    '''
    def __init__(self, in_planes, out_planes, expansion, layer_conf, stride, device, add_se = False, act = 'ReLU', bn = nn.BatchNorm2d , att_flag = False):
        super(MicroBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride =  layer_conf[1][1]
        planes = int(expansion * in_planes)

        # (expansion, out_planes, num_blocks, [[kernel, stride, padding ,bias],[kernel, stride, padding, bias],[kernel, stride, padding, bias], [kernel, stride, padding,bias], [kernel, bias])

        if act == 'HSwish':
            self.cba1 = ConvBNHSwish(in_planes, planes, kernel_size=layer_conf[0][0], stride=layer_conf[0][1], padding=layer_conf[0][2], bias=layer_conf[0][3], momentum=0.01, device=device, norm_layer = bn)
            self.cba2 = ConvBNHSwish(planes, planes, kernel_size=layer_conf[1][0], stride=layer_conf[1][1], padding=layer_conf[1][2], groups=planes, bias=layer_conf[1][3], momentum=0.01, device=device, norm_layer = bn)
        elif act == 'Mish':
            self.cba1 = ConvBNMish(in_planes, planes, kernel_size=layer_conf[0][0], stride=layer_conf[0][1], padding=layer_conf[0][2], bias=layer_conf[0][3], momentum=0.01, device=device, norm_layer = bn)
            self.cba2 = ConvBNMish(planes, planes, kernel_size=layer_conf[1][0], stride=layer_conf[1][1], padding=layer_conf[1][2], groups=planes, bias=layer_conf[1][3], momentum=0.01, device=device, norm_layer = bn)

        else:
            self.cba1 = ConvBNReLU(in_planes, planes, kernel_size=layer_conf[0][0], stride=layer_conf[0][1], padding=layer_conf[0][2], bias=layer_conf[0][3], momentum=0.01, device=device, norm_layer = bn)
            self.cba2 = ConvBNReLU(planes, planes, kernel_size=layer_conf[1][0], stride=layer_conf[1][1], padding=layer_conf[1][2], groups=planes, bias=layer_conf[1][3], momentum=0.01, device=device, norm_layer = bn)

        self.cb3 = ConvBN(planes, out_planes, kernel_size=layer_conf[2][0], stride=layer_conf[2][1], padding=layer_conf[2][2],bias=layer_conf[2][3], momentum=0.01, device=device, norm_layer = bn)

        self.shortcut = nn.Sequential()
        stride = layer_conf[2][1]
        if stride == 1 and in_planes != out_planes:
            self.shortcut = ConvBN(in_planes, out_planes, kernel_size=layer_conf[3][0], stride=layer_conf[3][1], padding=layer_conf[3][2], bias=layer_conf[3][3], momemtun=0.01, device=device, norm_layer = bn)


        # CBAM
        self.att_flag = att_flag
        if self.att_flag == True:
            self.ca = ChannelAttention(out_planes)
            self.sa = SpatialAttention()

        self.add_se = add_se
        if self.add_se:
            number = int(out_planes*0.25)
            self.se = SEModule(out_planes, number, kernel_size=layer_conf[4][0], bias=layer_conf[4][1], device=device, norm_layer = bn)  
        

    def forward(self, x):
        out = self.cba1(x)
        out = self.cba2(out)
        out = self.cb3(out)

        # CBAM
        if self.att_flag:
            out = self.ca(out) * out
            out = self.sa(out) * out
        
        # Squeeze-Excitation
        if self.add_se:
            if self.stride == 1:
                w = self.se(out)
                out = out * w + self.shortcut(x)
                return  out
            else:
                return  out

        if self.stride == 1:
            out = out + self.shortcut(x)
            return out 
        else:
            return out 


class MicroNet(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    def __init__(self, ver = 'ver1', ver_new_cfg = None, run_type='train', num_classes=100, wide_factor = 1, depth_factor =1, add_se = True, 
                       Activation = 'ReLU',
                       Batchnorm = 'BatchNorm',
                       device = 'cpu'):

        super(MicroNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        '''
        wide_factor: ratio to expand channel
        depth_factor: ratio to expand depth
        '''
        # (expansion, out_planes, num_blocks, stride)
        #                                                   cba1                           cba2                             cba3                    shortcut                      se
        # (expansion, out_planes, num_blocks, [[kernel, stride, padding ,bias],[kernel, stride, padding, bias],[kernel, stride, padding, bias], [kernel, stride, padding,bias], [kernel, bias])
        if ver == 'ver1':
            self.cfg = [[3, 16,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [3, 32,  1, [[1,1,0,False],[3,2,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [3, 32,  1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [3, 48,  3, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [3, 72,  1, [[1,1,0,False],[3,2,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [3, 72,  4, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [3, 80,  1, [[1,1,0,False],[3,2,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [3, 88,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [3, 106, 1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]]
                      ]
        elif ver == 'ver2':
            self.cfg = [[2.5, 20,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [2.5, 36,  1, [[1,1,0,False],[3,2,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [2.5, 36,  1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [2.5, 56,  3, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [2.5, 80,  1, [[1,1,0,False],[3,2,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [2.5, 80,  4, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [2.5, 88,  1, [[1,1,0,False],[3,2,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [2.5, 96,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]],
                        [2.5, 114, 1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,True]]]
                      ]
        elif ver == 'ver3': #imagenet-scaled
            self.cfg = [[1, 16,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 24,  1, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 24,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 40,  1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 40,  2, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 80,  1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 80,  2, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 96,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 192, 1, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 192, 3, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 320, 1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]]
                      ]
        #                                                   cba1                           cba2                             cba3                    shortcut                      se
        # (expansion, out_planes, num_blocks, [[kernel, stride, padding ,bias],[kernel, stride, padding, bias],[kernel, stride, padding, bias], [kernel, stride, padding,bias], [kernel, bias])
        elif ver == 'ver4' or ver == 'ver4-cbam': #large imagenet-scaled
            self.cfg = [[1, 16,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 24,  1, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 24,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 40,  1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 40,  2, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 80,  1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 80,  2, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [5, 96,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [5, 192, 1, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [5, 245, 3, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [5, 490, 1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]]
                      ]
        elif ver == 'ver5': #large imagenet-scaled
            self.cfg = [[1, 32,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 32,  1, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 64,  2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [3, 64,  1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [5, 96,  2, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [5, 96,  1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [5, 128, 2, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [7, 128, 2, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [7, 192, 1, [[1,1,0,False],[5,2,2,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [7, 245, 3, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]],
                        [7, 980, 1, [[1,1,0,False],[3,1,1,False],[1,1,0,False],[1,1,0,False],[1,False]]]
                      ]
        else:
            #print("MicroNet's new net_cfg:", ver_new_cfg)
            self.cfg = ver_new_cfg

        self.change_cfg(wide_factor, depth_factor)
        
        self.device = device

        self.run_type = run_type

        #construct network
        self.add_se = add_se
        self.act = Activation
        self.input_channel = 32
        self.num_classes = num_classes

        #construct batchnorm
        self.norm = Batchnorm
        if self.norm == 'GhostBatchNorm':
            self.bn = GhostBatchNorm
        else:
            self.bn =  BatchNorm

        # CBAM: Convolutional Block Attention Module
        if '-cbam' in ver:  # ex.) ver4:False, ver4-cbam:True
            self.cbam = True
        else:
            self.cbam = False

        self.cb = ConvBN(3, self.input_channel, kernel_size=3, stride=1, padding=1, bias=False, momentum=0.01, norm_layer = self.bn)

        self.layers = self._make_layers(in_planes=self.input_channel,run_type=self.run_type, att_flag = self.cbam)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.cfg[-1][1], self.num_classes)
        
        self.act = Activation
        if self.act == 'HSwish':
            self.stem_act = HSwish()
        elif self.act == 'Mish':
            self.stem_act = Mish()
        else:
            self.stem_act = nn.ReLU()
        
        #initialize the parameters
        self.reset_parameters()
        
        #initialize the parameters
        self.reset_custom_parameters()
        
        
    def forward(self, x):
        #stem
        out = self.cb(x)
        out = self.stem_act(out)
        out = self.layers(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.linear(self.dropout(out))
            
        return out

    def set_masks(self, masks):
        self.cb.set_mask(masks[0])
        number = 1
        for i, layer in enumerate(self.layers):
            layer.cba1.set_mask(masks[number])
            number += 1
            layer.cba2.set_mask(masks[number])
            number += 1
            layer.cb3.set_mask(masks[number])
            number += 1
            if layer.shortcut:
                layer.shortcut.set_mask(masks[number])
                number += 1
            if layer.add_se:
                layer.se.set_mask(masks[number], masks[number+1])
                number += 2
        self.linear.set_mask(masks[-1])

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    #m.bias = nn.Parameter(torch.zeros(m.running_var.shape, dtype=m.running_var.dtype, device=m.running_var.device), requires_grad=False)
                else:
                    if m.weight is None:
                        m.weight = nn.Parameter(torch.ones(m.running_var.shape, dtype=m.running_var.dtype, device=m.running_var.device), requires_grad=False)
                    if m.bias is None:
                        m.bias = nn.Parameter(torch.zeros(m.running_var.shape, dtype=m.running_var.dtype, device=m.running_var.device), requires_grad=False)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
             
    def reset_custom_parameters(self):
        for name, module in self.named_children():
            if 'layer' in name:
                if 'cba1' in name:
                    conv2d_module = module.cb[0]
                    n = conv2d_module.shape[1]
                    conv2d_module.data.normal_(0, 0.65 * math.sqrt(1. / n))
                elif 'cba2' in name:
                    conv2d_module = module.cb[0]
                    n = conv2d_module.shape[0]
                    conv2d_module.data.normal_(0, 9.37 * math.sqrt(2. / n))
                elif 'cb3' in name:
                    conv2d_module = module[0]
                    n = conv2d_module.shape[1]
                    conv2d_module.data.normal_(0, 1.18 * math.sqrt(1. / n))

    def _make_layers(self, in_planes, run_type, att_flag):
        layers = []
        for expansion, out_planes, num_blocks, layer_conf in self.cfg:
            strides = [int(layer_conf[3][1])] + [1]*(int(num_blocks)-1)
            for stride in strides:
                layers.append(MicroBlock(in_planes, out_planes, expansion, layer_conf, stride, self.device, self.add_se, self.act, self.bn, att_flag))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def change_cfg(self, wide_factor, depth_factor):
        for i in range(len(self.cfg)):
            self.cfg[i][1] = int(self.cfg[i][1] * wide_factor)
            if self.cfg[i][3] ==1:
                self.cfg[i][2] = int(self.cfg[i][2] * depth_factor)
    
    
    def set_config(self, batch_size, momentum, lr, num_epochs, device, weight_decay, gamma = 0.1, milestones = [100,150], nesterov = True, criterion = nn.CrossEntropyLoss()):
        self.batch_size = batch_size
        self.momentum = momentum
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.milestones = milestones
        self.device = device
        self.nesterov = nesterov
        

def micronet(conf, ver="ver1", run_type="train", device_type=None):

    if device_type == None:
        device = conf.get()['cuda']['device'] if conf.get()['cuda']['avail'] else torch.device('cpu')


    if conf.get()['model']['config'] != 'none':
        net = MicroNet(ver = "ver_new", 
                       ver_new_cfg = conf.get()['model']['config'], 
                       num_classes = conf.get()['model']['num_class'],
                       add_se = True, 
                       Activation = 'ReLU',         # { ReLU, HSwish }
                       Batchnorm = 'BatchNorm',  # { BatchNorm, GhostBatchNorm }
                       device = device_type)
    else:     
        net = MicroNet(ver = ver,  
                       run_type= run_type,
                       num_classes = conf.get()['model']['num_class'], 
                       add_se = True, 
                       Activation = 'Mish',          # { ReLU, HSwish, Mish }
                       Batchnorm = 'GhostBatchNorm',   # { BatchNorm, GhostBatchNorm }
                       device = device_type)

    net.set_config(batch_size = conf.get()['model']['batch'], 
                   momentum = conf.get()['optimizer']['momentum'], 
                   lr = conf.get()['model']['lr'], 
                   num_epochs =conf.get()['model']['training']['train_epochs'], 
                   weight_decay = conf.get()['optimizer']['weight_decay'], 
                   device = device_type,
                   nesterov = True)
    return net

#model = MicroNet(ver="ver4-cbam", num_classes=41, add_se = True, Activation='Mish', Batchnorm='GhostBatchNorm')
#print(model)
#out = model(torch.randn(1,3,224,224))
#torch.save(model.state_dict(), './tmp.p')

