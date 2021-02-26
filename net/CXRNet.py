# encoding: utf-8
"""
CXRNet for Abnormality Classification and Location 
Author: Jason.Fang
E-mail: fangjiansheng@cvte.com
Update time: 02/02/2021
"""
import sys
import re
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label as skmlabel
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

#define by myself
#sys.path.append("..") 
#from CXRAD.config import *
from config import *

#construct model
class CXRNet(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(CXRNet, self).__init__()
        self.sa = SpatialAttention()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        
    def forward(self, x):
        #x: N*C*W*H
        #x = self.sa(x) * x
        x = self.dense_net_121.features(x) #bz*1024*7*7
        x = self.sa(x) * x
        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return x, out
        
class SpatialAttention(nn.Module):#spatial attention layer
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    #for debug   
    x = torch.rand(10, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = CXRNet(num_classes = len(CLASS_NAMES_NIH), is_pre_trained=True)
    fea_conv, out = model(x)
    print(fea_conv.size())
    print(out.size())
