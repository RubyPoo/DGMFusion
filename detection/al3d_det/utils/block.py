import torch
from torch import nn
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable

import torch
from torch import nn
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable

from os.path import join as pjoin
from collections import OrderedDict

import torch.nn.functional as F



class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C,height, width = x.size()
        a = self.query_conv(x)
        #print("a.shape:\n", a.shape)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        #print("proj_query.shape:\n", proj_query.shape)
        
        b = self.key_conv(x)
        #print("b.shape:\n", b.shape)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        #print("proj_key.shape:\n", proj_key.shape)
        
        energy = torch.bmm(proj_query, proj_key)
        #print("energy.shape:\n", energy.shape)
        attention = self.softmax(energy)
        #print("attention.shape:\n", attention.shape)
        
        c = self.value_conv(x)
        #print("c.shape:\n", c.shape)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        #print("proj_value.shape:\n", proj_value.shape)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        #print("out.shape:\n", out.shape)
        out = out.view(m_batchsize, C, height, width)
        #print("out.shape:\n", out.shape)

        out = self.gamma*out + x
        #print("out.shape:\n", out.shape)
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        #print("x.shape:\n", x.shape)
        proj_query = x.view(m_batchsize, C, -1)
        #print("proj_query.shape:\n", proj_query.shape)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        #print("proj_key.shape:\n", proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)
        #print("energy.shape:\n", energy.shape)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        #print("energy_new.shape:\n", energy_new.shape)
        attention = self.softmax(energy_new)
        #print("attention.shape:\n", attention.shape)
        proj_value = x.view(m_batchsize, C, -1)
        #print("proj_value.shape:\n", proj_value.shape)

        out = torch.bmm(attention, proj_value)
        #print("out.shape:\n", out.shape)
        out = out.view(m_batchsize, C, height, width)
        #print("out.shape:\n", out.shape)

        out = self.gamma*out + x
        #print("out.shape:\n", out.shape)
        return out


def norm(planes, mode='bn', groups=16):
    if mode == 'bn':
        return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()

# de
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 16
#         inter_channels = in_channels  # test

        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        

    def forward(self, x):
#         x = x.unsqueeze(0) 
        #print("x.shape:\n", x.shape)
        feat1 = self.conv5a(x)
        #print("feat1.shape:\n", feat1.shape)
        sa_feat = self.sa(feat1)
        #print("sa_feat.shape:\n", sa_feat.shape)
        sa_conv = self.conv51(sa_feat)
        #print("sa_conv.shape:\n", sa_conv.shape)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        #print("feat2.shape:\n", feat2.shape)
        sc_feat = self.sc(feat2)
        #print("sc_feat.shape:\n", sc_feat.shape)
        sc_conv = self.conv52(sc_feat)
        #print("sc_conv.shape:\n", sc_conv.shape)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv
        #print("feat_sum.shape:\n", feat_sum.shape)

        sasc_output = self.conv8(feat_sum)
        #print("sasc_output.shape:\n", sasc_output.shape)
        

        return sasc_output
