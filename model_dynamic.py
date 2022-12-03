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


class MEF_dynamic(nn.Module):
    def __init__(self):
        super(MEF_dynamic, self).__init__()
        filters_in = 3
        filters_out = 3
        nFeat = 32

        # encoder1
        self.conv1 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv1_E0_1 = nn.Conv2d(nFeat, nFeat, 3, 2, 1, bias=True)
        self.conv1_E0_2 = Res_block(nFeat)
        self.conv1_E1_1 = nn.Conv2d(nFeat, nFeat * 2, 3, 2, 1, bias=True)
        self.conv1_E1_2 = Res_block(nFeat * 2)
        self.conv1_E2_1 = nn.Conv2d(nFeat * 2, nFeat * 4, 3, 2, 1, bias=True)
        self.conv1_E2_2 = Res_block(nFeat * 4)
        self.conv1_E3_1 = nn.Conv2d(nFeat * 4, nFeat * 8, 3, 2, 1, bias=True)
        self.conv1_E3_2 = Res_block(nFeat * 8)

        # encoder2
        self.conv2 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv2_E0_1 = nn.Conv2d(nFeat, nFeat, 3, 2, 1, bias=True)
        self.conv2_E0_2 = Res_block(nFeat)
        self.conv2_E1_1 = nn.Conv2d(nFeat, nFeat * 2, 3, 2, 1, bias=True)
        self.conv2_E1_2 = Res_block(nFeat * 2)
        self.conv2_E2_1 = nn.Conv2d(nFeat * 2, nFeat * 4, 3, 2, 1, bias=True)
        self.conv2_E2_2 = Res_block(nFeat * 4)
        self.conv2_E3_1 = nn.Conv2d(nFeat * 4, nFeat * 8, 3, 2, 1, bias=True)
        self.conv2_E3_2 = Res_block(nFeat * 8)

        # merge
        self.conv_merge = nn.Conv2d(nFeat * 16, nFeat * 8, 3, 1, 1, bias=True)
        self.res_module = Res_blocks(nFeat * 8, 3)

        # decoder
        self.conv_D3_1 = nn.ConvTranspose2d(nFeat * 8, nFeat * 4, 4, 2, 1, bias=True)
        self.conv_D3_2 = Res_block(nFeat * 4)
        self.conv_D2_1 = nn.ConvTranspose2d(nFeat * 12, nFeat * 2, 4, 2, 1, bias=True)
        self.conv_D2_2 = Res_block(nFeat * 2)
        self.conv_D1_1 = nn.ConvTranspose2d(nFeat * 6, nFeat, 4, 2, 1, bias=True)
        self.conv_D1_2 = Res_block(nFeat)
        self.conv_D0_1 = nn.ConvTranspose2d(nFeat * 3, nFeat, 4, 2, 1, bias=True)
        self.conv_D0_2 = Res_block(nFeat)
        self.conv_D_1 = nn.Conv2d(nFeat * 3, nFeat, 3, 1, 1, bias=True)
        self.conv_D_2 = Res_block(nFeat)

        self.conv_out = nn.Conv2d(nFeat, filters_out, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, imgs):
        img1 = imgs[:, :3, :, :]
        img2 = imgs[:, 3:6, :, :]

        # Encoder1
        E1 = self.relu(self.conv1(img1))
        E1_0 = self.relu(self.conv1_E0_1(E1))
        E1_0 = self.conv1_E0_2(E1_0)
        E1_1 = self.relu(self.conv1_E1_1(E1_0))
        E1_1 = self.conv1_E1_2(E1_1)
        E1_2 = self.relu(self.conv1_E2_1(E1_1))
        E1_2 = self.conv1_E2_2(E1_2)
        E1_3 = self.relu(self.conv1_E3_1(E1_2))
        E1_3 = self.conv1_E3_2(E1_3)

        # Encoder2
        E2 = self.relu(self.conv2(img2))
        E2_0 = self.relu(self.conv2_E0_1(E2))
        E2_0 = self.conv2_E0_2(E2_0)
        E2_1 = self.relu(self.conv2_E1_1(E2_0))
        E2_1 = self.conv2_E1_2(E2_1)
        E2_2 = self.relu(self.conv2_E2_1(E2_1))
        E2_2 = self.conv2_E2_2(E2_2)
        E2_3 = self.relu(self.conv2_E3_1(E2_2))
        E2_3 = self.conv2_E3_2(E2_3)

        fea_merged = self.relu(self.conv_merge(torch.cat([E1_3, E2_3], dim=1)))
        res_tensor = self.res_module(fea_merged)

        # Decoder
        D3 = self.relu(self.conv_D3_1(res_tensor))
        D3 = self.conv_D3_2(D3)
        D2 = self.relu(self.conv_D2_1(torch.cat([D3, E1_2, E2_2], dim=1)))
        D2 = self.conv_D2_2(D2)
        D1 = self.relu(self.conv_D1_1(torch.cat([D2, E1_1, E2_1], dim=1)))
        D1 = self.conv_D1_2(D1)
        D0 = self.relu(self.conv_D0_1(torch.cat([D1, E1_0, E2_0], dim=1)))
        D0 = self.conv_D0_2(D0)
        D = self.relu(self.conv_D_1(torch.cat([D0, E1, E2], dim=1)))
        D = self.conv_D_2(D)

        out = self.sigmoid(self.conv_out(D))
        #return D3, D2, D1, D0, D, out # for training
        return out
