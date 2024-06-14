# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn
from models.common import Conv,C3, Classify, SE_Block
from utils.downloads import attempt_download
import matplotlib.pyplot as plt
# from utils.general import select_device, LOGGER, check_img_size, increment_path
from utils.torch_utils import select_device, smart_inference_mode


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class EnsembleValidator(nn.Module):

    def __init__(self, weights, device=None, fp16=False, batch_size=64):
        super(EnsembleValidator, self).__init__()
        self.device = select_device(device, batch_size=batch_size)
        self.stride = 32
        self.pt = True
        self.jit = False
        self.engine = False
        self.fp16 = fp16
        self.batch_size = batch_size
        self.weights = weights


        self.model = self.load_model(weights)

    def load_model(self, weights):

        if isinstance(weights, list):
            weights = weights[0]

        fusion = FusionComponent(3,3,8)
        model = Ensemble(fusion)

        checkpoint = torch.load(weights, map_location=self.device)
        model_state_dict = checkpoint['model']
        model.load_state_dict(model_state_dict)
        return model

    def forward(self, x1, x2):

        return self.model(x1, x2)

class FusionComponent(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FusionComponent, self).__init__()

        self.conv_mask1 = nn.Conv2d(in_channels1, 1, 1, 1, 0, bias=True)
        self.conv_mask2 = nn.Conv2d(in_channels2, 1, 1, 1, 0, bias=True)
        self.conv_mask3 = nn.Conv2d(in_channels1, 1, 1, 1, 0, bias=True)
        self.conv_mask4 = nn.Conv2d(in_channels2, 1, 1, 1, 0, bias=True)

        self.conv1 = nn.Conv2d(in_channels=in_channels1, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=in_channels2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.se_1 = SE_Block(out_channels, 16)
        self.se_2 = SE_Block(out_channels, 16)

        self.se = SE_Block(2*out_channels, 16)
        self.conv_fuse = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.Silu = nn.SiLU()

        #
        # self.conv3_0 = Conv(c1=out_channels, c2=out_channels, k=3, s=1, p=1)
        self.conv3_1 = Conv(c1=2 *out_channels, c2=32, k=6, s=2, p=2)
        self.conv3_2 = Conv(c1=32, c2=64, k=3, s=2, p=1)
        self.c3_1 = C3(c1 = 64, c2 = 64)
        self.conv3_3 = Conv(c1=64, c2=128, k=3, s=2, p=1)
        self.c3_2 = C3(c1=128, c2=128)
        self.conv3_4 = Conv(c1=128, c2=256, k=3, s=2, p=1)
        self.c3_3 = C3(c1=256, c2=256)
        self.conv3_5 = Conv(c1=256, c2=512, k=3, s=2, p=1)
        self.c3_4 = C3(c1=512, c2=512)
        self.Classify = Classify(c1=512, c2=19)
        self.stride = 32
    def forward(self, x1, x2):

        x1 = x1*0.5
        x2 = x2*0.5


        x1_masked = torch.mul(self.conv_mask1(x1).repeat(1, 3, 1, 1), x1)
        x2_masked = torch.mul(self.conv_mask2(x2).repeat(1, 3, 1, 1), x2)



        mask_x1 = self.se_1(self.Silu(self.bn1(self.conv1(x1_masked + x1))))
        mask_x2 = self.se_2(self.Silu(self.bn2(self.conv2(x2_masked + x2))))

        x = self.se(torch.cat([mask_x1, mask_x2], 1))



        x = self.c3_1(self.conv3_2(self.conv3_1(x)))
        x = self.c3_2(self.conv3_3(x))
        x = self.c3_3(self.conv3_4(x))
        x = self.c3_4(self.conv3_5(x))
        x = self.Classify(x)

        return x



class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self,fusion_component):
        super().__init__()
        self.fusion_component = fusion_component

    def forward(self, x1,x2, augment=False, profile=False, visualize=False):

        return self.fusion_component(x1, x2)



def new_forward(self, x1, x2):
    x = fusion_component(x1, x2)
    x = original_forward(x)
    return x

def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model

    fusion_component = FusionComponent(in_channels1=3, in_channels2=3, out_channels=8)
    model = Ensemble(fusion_component=fusion_component)


    return model
