import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------
#  Convolutional Block
# -----------------------------------
class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.BatchNorm2d(64)

    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + identity

# -----------------------------------
#  Channel Attention Layer
# -----------------------------------
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.se(y)
        return x * y

# -----------------------------------
#  Spatial Attention Layer
# -----------------------------------
class SALayer(nn.Module):
    def __init__(self, channel):
        super(SALayer, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.sa(x)
        return x * y

# -----------------------------------
#  Squeeze-and-Excitation (SE) Layer
# -----------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# -----------------------------------
#  Residual Block with SE
# -----------------------------------
class Residual_Block_New(nn.Module):
    def __init__(self, in_num, out_num, dilation_factor):
        super(Residual_Block_New, self).__init__()
        self.conv1 = nn.Conv2d(in_num, out_num, 3, padding=dilation_factor, dilation=dilation_factor, bias=False)
        self.in1   = nn.BatchNorm2d(out_num)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_num, out_num, 3, padding=dilation_factor, dilation=dilation_factor, bias=False)
        self.in2   = nn.BatchNorm2d(out_num)
        self.se    = SELayer(out_num)

    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = self.se(out)
        return out + identity

class Residual_Block_Enhance(Residual_Block_New):
    def __init__(self, in_num, out_num, dilation_factor):
        super(Residual_Block_Enhance, self).__init__(in_num, out_num, dilation_factor)

# -----------------------------------
#  Residual Block with fixed dilation
# -----------------------------------
class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.in1   = nn.InstanceNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.in2   = nn.InstanceNorm2d(64)
        self.se    = SELayer(64)

    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = self.se(out)
        return out + identity

# -----------------------------------
#  Dense and Residual Dense Blocks
# -----------------------------------
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size,
                              padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return torch.cat([x, out], dim=1)

class RDB(nn.Module):
    def __init__(self, nChannels=64, nDenselayer=5, growthRate=16):
        super(RDB, self).__init__()
        channels = nChannels
        layers = []
        for _ in range(nDenselayer):
            layers.append(make_dense(channels, growthRate))
            channels += growthRate
        self.dense_layers = nn.Sequential(*layers)
        self.conv1x1 = nn.Conv2d(channels, nChannels, kernel_size=1, bias=False)
        self.se = SELayer(nChannels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out = self.se(out)
        return out + x
