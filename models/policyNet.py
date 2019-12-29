import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as torchinit
import math
from torch.nn import init, Parameter
import copy
import time
from functools import wraps

from models import base
import utils

class PolicyNet(nn.Module):

    def __init__(self, branch_num=3):
        super(PolicyNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=6, stride=3, padding=2, bias=False) #75*75
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    #38*38
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=1)  #12*12   #25*25
        # self.bn2 = nn.BatchNorm2d(64)
        # # self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=3, padding=0)  #8*8        #nn.Linear(resmodel.fc.in_features, int(hash_length)),nn.Sigmoid()
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  #6*6
        # self.bn3 = nn.BatchNorm2d(128)
        # # self.conv4 = nn.Conv2d(128, 128, kernel_size=6, stride=1, padding=0)
        # self.conv4 = nn.AvgPool2d(6)
        # self.fc1 = nn.Linear(128, size_num)
        # self.fc2 = nn.Linear(128, branch_num)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # self.fc1 = nn.Linear(512, size_num)
        # self.fc2 = nn.Linear(512, branch_num)
        self.fc = nn.Linear(512, branch_num)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.size())

        out = self.maxpool(out)
        # print(out.size())

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print(out.size())

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        # print(out.size())

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(-1,512)
        # print(out.size())

        # probs_size = F.softmax(self.fc1(out))
        # probs_branch = F.softmax(self.fc2(out))
        probs = F.softmax(self.fc(out))
        return probs
    # def forward(self, x):
    #
    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #     # print(out.size())
    #
    #     out = self.maxpool(out)
    #     # print(out.size())
    #
    #     out = self.conv2(out)
    #     out = self.bn2(out)
    #     out = self.relu(out)
    #     # print(out.size())
    #
    #     out = self.conv3(out)
    #     out = self.bn3(out)
    #     out = self.relu(out)
    #     # print(out.size())
    #
    #     out = self.conv4(out)
    #     out = self.relu(out)
    #     out = out.view(-1,128)
    #     # print(out.size())
    #
    #     probs_size = F.softmax(self.fc1(out))
    #     probs_branch = F.softmax(self.fc2(out))
    #
    #     return probs_size,probs_branch
    #
    # def forward_size(self, x):
    #     out = self.conv1(x)
    #     # out = self.bn1(out)
    #     out = self.relu(out)
    #
    #     out = self.maxpool(out)
    #
    #     out = self.conv2(out)
    #     out = self.relu(out)
    #
    #     out = self.conv3(out)
    #     out = self.relu(out)
    #
    #     out = self.conv4(out)
    #     out = self.relu(out)
    #
    #     out = F.softmax(self.fc1(out))
    #
    #     return out
    #
    # def forward_branch(self, x):
    #     out = self.conv1(x)
    #     # out = self.bn1(out)
    #     out = self.relu(out)
    #
    #     out = self.maxpool(out)
    #
    #     out = self.conv2(out)
    #     out = self.relu(out)
    #
    #     out = self.conv3(out)
    #     out = self.relu(out)
    #
    #     out = self.conv4(out)
    #     out = self.relu(out)
    #
    #     out = F.softmax(self.fc2(out))
    #
    #     return out
class PolicyNet_C10(nn.Module):

    def __init__(self, branch_num=3):
        super(PolicyNet_C10, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, branch_num)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # print(out.size())

        out = self.maxpool(out)
        # print(out.size())

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print(out.size())

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        # print(out.size())

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(-1,512)
        # print(out.size())

        # probs_size = F.softmax(self.fc1(out))
        # probs_branch = F.softmax(self.fc2(out))
        probs = F.softmax(self.fc(out))
        return probs

class PolicyNet_logits(nn.Module):

    def __init__(self, branch_num=3):
        super(PolicyNet_logits, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, branch_num)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.nn.Dropout(0.5)(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = torch.nn.Dropout(0.5)(out)
        out = self.relu(out)
        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(-1,512)
        logits = self.fc(out)

        return logits

class RgressionNet(nn.Module):

    def __init__(self, bits=10):
        super(RgressionNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, bits)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.nn.Dropout(0.5)(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = torch.nn.Dropout(0.5)(out)
        out = self.relu(out)
        out = nn.AvgPool2d(out.size(2))(out)
        out = out.view(-1,512)
        logits = self.fc(out)

        return logits