import torch
import torch.nn as nn
import torch.nn.functional as F


class Res_block(nn.Module):
    def __init__(self, nFeat, kernel_size=3):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out) + x
        return out


class Res_blocks(nn.Module):
    def __init__(self, nFeat, nReslayer):
        super(Res_blocks, self).__init__()
        modules = []
        for i in range(nReslayer):
            modules.append(Res_block(nFeat))
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out


class model_PE(nn.Module):
    def __init__(self):
        super(model_PE, self).__init__()
        filters_in = 3
        filters_out = 3
        nFeat = 32

        self.conv = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.res = Res_blocks(nFeat, 3)
        self.conv_out = nn.Conv2d(nFeat, filters_out, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.relu(self.conv(img))
        out = self.res(out)
        out = self.sigmoid(self.conv_out(out))
        return out


class model_PE_LH(nn.Module):
    def __init__(self):
        super(model_PE_LH, self).__init__()
        self.model_L = model_PE()
        self.model_H = model_PE()

    def forward(self, img_L, img_H):
        img_L = self.model_L(img_L)
        img_H = self.model_H(img_H)
        return img_L, img_H
