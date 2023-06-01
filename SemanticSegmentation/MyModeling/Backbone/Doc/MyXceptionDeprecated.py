import numpy as np
import torch
import torch.nn as nn
import h5py
from torchsummary import summary


class Xception(nn.Module):
    def __init__(self, modelPath='xception_weights_tf_dim_ordering_tf_kernels.h5'):
        super().__init__()
        self.shortcut = {}
        self.block = {}
        self.modelPath = 'xception_weights_tf_dim_ordering_tf_kernels.h5'
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(128)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), groups=64, bias=False, padding=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), groups=128, bias=False, padding=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(256)
        )
        self.block3 = self.separableConv(128, 256)
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(256, 728, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(728)
        )
        self.block4 = self.separableConv(256, 728)
        self.block5 = self.separableConv1(728)
        self.block6 = self.separableConv1(728)
        self.block7 = self.separableConv1(728)
        self.block8 = self.separableConv1(728)
        self.block9 = self.separableConv1(728)
        self.block10 = self.separableConv1(728)
        self.block11 = self.separableConv1(728)
        self.block12 = self.separableConv1(728)
        self.block13 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(728, 728, kernel_size=(3, 3), groups=728, bias=False, padding=(1, 1)),
            nn.Conv2d(728, 728, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            nn.Conv2d(728, 728, kernel_size=(3, 3), groups=728, bias=False, padding=(1, 1)),
            nn.Conv2d(728, 1024, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        self.shortcut13 = nn.Sequential(
            nn.Conv2d(728, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(1024)
        )
        self.block14 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), groups=1024, bias=False, padding=(1, 1)),
            nn.Conv2d(1024, 1536, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True)
        )
        self.block15 = nn.Sequential(
            nn.Conv2d(1536, 1536, kernel_size=(3, 3), groups=1536, bias=False, padding=(1, 1)),
            nn.Conv2d(1536, 2048, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.replaceWeight()

    @staticmethod
    def separableConv(inChannel, outChannel):
        block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inChannel, inChannel, kernel_size=(3, 3), groups=inChannel, bias=False, padding=(1, 1)),
            nn.Conv2d(inChannel, outChannel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannel, outChannel, kernel_size=(3, 3), groups=outChannel, bias=False, padding=(1, 1)),
            nn.Conv2d(outChannel, outChannel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(outChannel),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )

        return block

    @staticmethod
    def separableConv1(channel):
        block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), groups=channel, bias=False, padding=(1, 1)),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), groups=channel, bias=False, padding=(1, 1)),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), groups=channel, bias=False, padding=(1, 1)),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel)
        )

        return block

    def replaceWeight(self):
        self.block['block1'] = self.block1
        self.block['block2'] = self.block2
        self.block['block3'] = self.block3
        self.block['block4'] = self.block4
        self.block['block5'] = self.block5
        self.block['block6'] = self.block6
        self.block['block7'] = self.block7
        self.block['block8'] = self.block8
        self.block['block9'] = self.block9
        self.block['block10'] = self.block10
        self.block['block11'] = self.block11
        self.block['block12'] = self.block12
        self.block['block13'] = self.block13
        self.block['block14'] = self.block14
        self.block['block15'] = self.block15
        self.shortcut['shortcut1'] = self.shortcut2
        self.shortcut['shortcut2'] = self.shortcut3
        self.shortcut['shortcut3'] = self.shortcut4
        self.shortcut['shortcut4'] = self.shortcut13
        with h5py.File(self.modelPath, 'r') as f:
            self.block['block1'][0].weight.data = torch.Tensor(
                np.transpose(f['/convolution2d_1']['convolution2d_1_W:0'][:], axes=(3, 2, 0, 1)))
            self.block['block1'][3].weight.data = torch.Tensor(
                np.transpose(f['/convolution2d_2']['convolution2d_2_W:0'][:], axes=(3, 2, 0, 1)))
            for i in range(1, 5):
                self.shortcut['shortcut' + str(i)][0].weight.data = torch.Tensor(
                    np.transpose(f['/convolution2d_' + str(i + 2)]['convolution2d_' + str(i + 2) + '_W:0'][:],
                                 axes=(3, 2, 0, 1)))

            self.block['block2'][0].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_1']['separableconvolution2d_1_depthwise_kernel:0'][:],
                             axes=(2, 3, 0, 1)))
            self.block['block2'][1].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_1']['separableconvolution2d_1_pointwise_kernel:0'][:],
                             axes=(3, 2, 0, 1)))
            self.block['block2'][4].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_2']['separableconvolution2d_2_depthwise_kernel:0'][:],
                             axes=(2, 3, 0, 1)))
            self.block['block2'][5].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_2']['separableconvolution2d_2_pointwise_kernel:0'][:],
                             axes=(3, 2, 0, 1)))
            for i in range(1, 3):
                self.block['block' + str(i + 2)][1].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 2 + 1)][
                                     'separableconvolution2d_' + str(i * 2 + 1) + '_depthwise_kernel:0'][:],
                                 axes=(2, 3, 0, 1)))
                self.block['block' + str(i + 2)][2].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 2 + 1)][
                                     'separableconvolution2d_' + str(i * 2 + 1) + '_pointwise_kernel:0'][:],
                                 axes=(3, 2, 0, 1)))
                self.block['block' + str(i + 2)][5].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 2 + 2)][
                                     'separableconvolution2d_' + str(i * 2 + 2) + '_depthwise_kernel:0'][:],
                                 axes=(2, 3, 0, 1)))
                self.block['block' + str(i + 2)][6].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 2 + 2)][
                                     'separableconvolution2d_' + str(i * 2 + 2) + '_pointwise_kernel:0'][:],
                                 axes=(3, 2, 0, 1)))
            for i in range(1, 9):
                self.block['block' + str(i + 4)][1].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 3 + 4)][
                                     'separableconvolution2d_' + str(i * 3 + 4) + '_depthwise_kernel:0'][:],
                                 axes=(2, 3, 0, 1)))
                self.block['block' + str(i + 4)][2].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 3 + 4)][
                                     'separableconvolution2d_' + str(i * 3 + 4) + '_pointwise_kernel:0'][:],
                                 axes=(3, 2, 0, 1)))
                self.block['block' + str(i + 4)][5].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 3 + 5)][
                                     'separableconvolution2d_' + str(i * 3 + 5) + '_depthwise_kernel:0'][:],
                                 axes=(2, 3, 0, 1)))
                self.block['block' + str(i + 4)][6].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 3 + 5)][
                                     'separableconvolution2d_' + str(i * 3 + 5) + '_pointwise_kernel:0'][:],
                                 axes=(3, 2, 0, 1)))
                self.block['block' + str(i + 4)][9].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 3 + 6)][
                                     'separableconvolution2d_' + str(i * 3 + 6) + '_depthwise_kernel:0'][:],
                                 axes=(2, 3, 0, 1)))
                self.block['block' + str(i + 4)][10].weight.data = torch.Tensor(
                    np.transpose(f['/separableconvolution2d_' + str(i * 3 + 6)][
                                     'separableconvolution2d_' + str(i * 3 + 6) + '_pointwise_kernel:0'][:],
                                 axes=(3, 2, 0, 1)))
            self.block['block' + str(13)][1].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_' + str(31)][
                                 'separableconvolution2d_' + str(31) + '_depthwise_kernel:0'][:],
                             axes=(2, 3, 0, 1)))
            self.block['block' + str(13)][2].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_' + str(31)][
                                 'separableconvolution2d_' + str(31) + '_pointwise_kernel:0'][:],
                             axes=(3, 2, 0, 1)))
            self.block['block' + str(13)][5].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_' + str(32)][
                                 'separableconvolution2d_' + str(32) + '_depthwise_kernel:0'][:],
                             axes=(2, 3, 0, 1)))
            self.block['block' + str(13)][6].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_' + str(32)][
                                 'separableconvolution2d_' + str(32) + '_pointwise_kernel:0'][:],
                             axes=(3, 2, 0, 1)))
            self.block['block' + str(14)][0].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_' + str(33)][
                                 'separableconvolution2d_' + str(33) + '_depthwise_kernel:0'][:],
                             axes=(2, 3, 0, 1)))
            self.block['block' + str(14)][1].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_' + str(33)][
                                 'separableconvolution2d_' + str(33) + '_pointwise_kernel:0'][:],
                             axes=(3, 2, 0, 1)))
            self.block['block' + str(15)][0].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_' + str(34)][
                                 'separableconvolution2d_' + str(34) + '_depthwise_kernel:0'][:],
                             axes=(2, 3, 0, 1)))
            self.block['block' + str(15)][1].weight.data = torch.Tensor(
                np.transpose(f['/separableconvolution2d_' + str(34)][
                                 'separableconvolution2d_' + str(34) + '_pointwise_kernel:0'][:],
                             axes=(3, 2, 0, 1)))
            self.block['block1'][1].weight.data = torch.Tensor(
                f['/batchnormalization_1']['batchnormalization_1_gamma:0'][:])
            self.block['block1'][1].bias.data = torch.Tensor(
                f['/batchnormalization_1']['batchnormalization_1_beta:0'][:])
            self.block['block1'][4].weight.data = torch.Tensor(
                f['/batchnormalization_2']['batchnormalization_2_gamma:0'][:])
            self.block['block1'][4].bias.data = torch.Tensor(
                f['/batchnormalization_2']['batchnormalization_2_beta:0'][:])
            self.shortcut['shortcut1'][1].weight.data = torch.Tensor(
                f['/batchnormalization_3']['batchnormalization_3_gamma:0'][:])
            self.shortcut['shortcut1'][1].bias.data = torch.Tensor(
                f['/batchnormalization_3']['batchnormalization_3_beta:0'][:])
            self.block['block2'][2].weight.data = torch.Tensor(
                f['/batchnormalization_4']['batchnormalization_4_gamma:0'][:])
            self.block['block2'][2].bias.data = torch.Tensor(
                f['/batchnormalization_4']['batchnormalization_4_beta:0'][:])
            self.block['block2'][6].weight.data = torch.Tensor(
                f['/batchnormalization_5']['batchnormalization_5_gamma:0'][:])
            self.block['block2'][6].bias.data = torch.Tensor(
                f['/batchnormalization_5']['batchnormalization_5_beta:0'][:])
            self.shortcut['shortcut2'][1].weight.data = torch.Tensor(
                f['/batchnormalization_6']['batchnormalization_6_gamma:0'][:])
            self.shortcut['shortcut2'][1].bias.data = torch.Tensor(
                f['/batchnormalization_6']['batchnormalization_6_beta:0'][:])
            self.block['block3'][3].weight.data = torch.Tensor(
                f['/batchnormalization_7']['batchnormalization_7_gamma:0'][:])
            self.block['block3'][3].bias.data = torch.Tensor(
                f['/batchnormalization_7']['batchnormalization_7_beta:0'][:])
            self.block['block3'][7].weight.data = torch.Tensor(
                f['/batchnormalization_8']['batchnormalization_8_gamma:0'][:])
            self.block['block3'][7].bias.data = torch.Tensor(
                f['/batchnormalization_8']['batchnormalization_8_beta:0'][:])
            self.shortcut['shortcut3'][1].weight.data = torch.Tensor(
                f['/batchnormalization_9']['batchnormalization_9_gamma:0'][:])
            self.shortcut['shortcut3'][1].bias.data = torch.Tensor(
                f['/batchnormalization_9']['batchnormalization_9_beta:0'][:])
            self.block['block4'][3].weight.data = torch.Tensor(
                f['/batchnormalization_10']['batchnormalization_10_gamma:0'][:])
            self.block['block4'][3].bias.data = torch.Tensor(
                f['/batchnormalization_10']['batchnormalization_10_beta:0'][:])
            self.block['block4'][7].weight.data = torch.Tensor(
                f['/batchnormalization_11']['batchnormalization_11_gamma:0'][:])
            self.block['block4'][7].bias.data = torch.Tensor(
                f['/batchnormalization_11']['batchnormalization_11_beta:0'][:])
            for i in range(1, 9):
                self.block['block' + str(i + 4)][3].weight.data = torch.Tensor(
                    f['/batchnormalization_' + str(i * 3 + 9)]['batchnormalization_' + str(i * 3 + 9) + '_gamma:0'][:])
                self.block['block' + str(i + 4)][3].bias.data = torch.Tensor(
                    f['/batchnormalization_' + str(i * 3 + 9)]['batchnormalization_' + str(i * 3 + 9) + '_beta:0'][:])
                self.block['block' + str(i + 4)][7].weight.data = torch.Tensor(
                    f['/batchnormalization_' + str(i * 3 + 10)]['batchnormalization_' + str(i * 3 + 10) + '_gamma:0'][
                    :])
                self.block['block' + str(i + 4)][7].bias.data = torch.Tensor(
                    f['/batchnormalization_' + str(i * 3 + 10)]['batchnormalization_' + str(i * 3 + 10) + '_beta:0'][:])
                self.block['block' + str(i + 4)][11].weight.data = torch.Tensor(
                    f['/batchnormalization_' + str(i * 3 + 11)]['batchnormalization_' + str(i * 3 + 11) + '_gamma:0'][
                    :])
                self.block['block' + str(i + 4)][11].bias.data = torch.Tensor(
                    f['/batchnormalization_' + str(i * 3 + 11)]['batchnormalization_' + str(i * 3 + 11) + '_beta:0'][:])

            self.block['block13'][7].weight.data = torch.Tensor(
                f['/batchnormalization_36']['batchnormalization_36_gamma:0'][:])
            self.block['block13'][7].bias.data = torch.Tensor(
                f['/batchnormalization_36']['batchnormalization_36_beta:0'][:])
            self.block['block13'][3].weight.data = torch.Tensor(
                f['/batchnormalization_37']['batchnormalization_37_gamma:0'][:])
            self.block['block13'][3].bias.data = torch.Tensor(
                f['/batchnormalization_37']['batchnormalization_37_beta:0'][:])
            self.shortcut['shortcut4'][1].weight.data = torch.Tensor(
                f['/batchnormalization_38']['batchnormalization_38_gamma:0'][:])
            self.shortcut['shortcut4'][1].bias.data = torch.Tensor(
                f['/batchnormalization_38']['batchnormalization_38_beta:0'][:])
            self.block['block14'][2].weight.data = torch.Tensor(
                f['/batchnormalization_39']['batchnormalization_39_gamma:0'][:])
            self.block['block14'][2].bias.data = torch.Tensor(
                f['/batchnormalization_39']['batchnormalization_39_beta:0'][:])
            self.block['block15'][2].weight.data = torch.Tensor(
                f['/batchnormalization_40']['batchnormalization_40_gamma:0'][:])
            self.block['block15'][2].bias.data = torch.Tensor(
                f['/batchnormalization_40']['batchnormalization_40_beta:0'][:])

    def forward(self, input):  # 3,299,299
        s1 = self.block1(input)  # 64,147,147
        s2 = self.block2(s1) + self.shortcut2(s1)  # 128,74,74
        s3 = self.block3(s2) + self.shortcut3(s2)  # 256,37,37
        s4 = self.block4(s3) + self.shortcut4(s3)  # 728,19,19
        s5 = self.block5(s4) + s4
        s6 = self.block6(s5) + s5
        s7 = self.block7(s6) + s6
        s8 = self.block8(s7) + s7
        s9 = self.block9(s8) + s8
        s10 = self.block10(s9) + s9
        s11 = self.block11(s10) + s10
        s12 = self.block12(s11) + s11  # 728,19,19
        s13 = self.block13(s12) + self.shortcut13(s12)  # 1024,10,10
        s14 = self.block14(s13)  # 1536,10,10
        s15 = self.block15(s14)  # 2048,10,10

        return s15


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=", device)
    xcep = Xception()
    xcep = xcep.to(device)
    print(summary(xcep, (3, 299, 299), batch_size=1))
    print(xcep)
    parm = {}
    for name, parameters in xcep.named_parameters():
        print(name, ':', parameters.size())
    # new_model = nn.Sequential(*list(xcep.children()))
    # print(new_model)
