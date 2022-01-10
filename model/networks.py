import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import copy
import sys
sys.path.append(('../'))
sys.path.append('../../')

'''
# ------------------------------------------------
# original file is kernelGAN-FKP
# ------------------------------------------------
'''

from utils.involution_cuda import involution

# ----------------------------------
# Networks of ZSSR INV
# ----------------------------------
class ZSSR_INV(nn.Module):

    def __init__(self, conf):
        super(ZSSR_INV, self).__init__()
        self.mid_channels = conf.mid_channels
        self.conv2_stride = conf.inv_stride

        self.conv_in = nn.Conv2d(in_channels= 3, out_channels= self.mid_channels, padding=2, kernel_size=3)

        feature_layer = []
        for _ in range(1, conf.n_layers-1):
            feature_layer +=[involution(self.mid_channels, 7, self.conv2_stride), nn.ReLU(inplace=True)]
            # self.bn = nn.BatchNorm2d(self.mid_channels)
        self.feature_layer = nn.Sequential(*feature_layer)
        self.conv_out = nn.Conv2d(in_channels= self.mid_channels, out_channels=3*4, kernel_size=3)
        self.pixel_out = nn.PixelShuffle(upscale_factor=2)



    def forward(self, input_tensor):
        input_feature = self.conv_in(input_tensor)

        feature = self.feature_layer(input_feature)
        pixel = self.conv_out(feature)
        output = self.pixel_out(pixel)

        return output




class ZSSR_ORI(nn.Module):

    def __init__(self, conf):
        super(ZSSR_ORI, self).__init__()
        self.mid_channels = conf.mid_channels
        self.conv2_stride = conf.inv_stride

        self.conv_in = nn.Conv2d(in_channels= 3, out_channels= self.mid_channels, padding=2, kernel_size=3)

        feature_layer = []
        for _ in range(1, conf.n_layers-1):
            feature_layer +=[nn.Conv2d(in_channels = self.mid_channels, out_channels=self.mid_channels, kernel_size=3), nn.ReLU(inplace=True)]
            # self.bn = nn.BatchNorm2d(self.mid_channels)
        self.feature_layer = nn.Sequential(*feature_layer)
        self.conv_out = nn.Conv2d(in_channels= self.mid_channels, out_channels=3*4, kernel_size=3)
        self.pixel_out = nn.PixelShuffle(upscale_factor=2)



    def forward(self, input_tensor):
        input_feature = self.conv_in(input_tensor)

        feature = self.feature_layer(input_feature)
        pixel = self.conv_out(feature)
        output = self.pixel_out(pixel)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight)
    # m.weight.data.normal_(0.0, 0.02)

