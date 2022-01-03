import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import copy
import sys
sys.path.append(('../'))
from util import swap_axis
sys.path.append('../../')
from FKP.network import KernelPrior

'''
# ------------------------------------------------
# original file is kernelGAN-FKP
# ------------------------------------------------
'''

from .utils.involution_cuda import involution

# ----------------------------------
# Networks of ZSSR INV
# ----------------------------------
class ZSSR_INV(nn.Module):

    def __init__(self, conf):
        super(ZSSR_INV, self).__init__()

        self.conv_in = nn.Conv2D(in_channels= 3, out_channels= self.mid_channels, kernel_size=3)

        self.involution = involution(self.mid_channels, 7, self.conv2_stride)
        self.bn = nn.BatchNorm2d(self.mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_out = nn.Conv2D(in_channels= self.mid_channels, out_channels=self.mid_channels*2, kernel_size=3)



    def forward(self, input_tensor):
