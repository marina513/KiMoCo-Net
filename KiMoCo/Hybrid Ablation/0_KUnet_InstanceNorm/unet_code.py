
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np



class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            (nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.InstanceNorm2d(ch_out,track_running_stats=False),
            nn.ReLU(inplace=True),
            (nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.InstanceNorm2d(ch_out,track_running_stats=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            (nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.InstanceNorm2d(ch_out,track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x




class U_Net_Complex_Real(nn.Module):
    def __init__(self, dim):
        super(U_Net_Complex_Real, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=dim, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # encoding path
        x1 = self.Conv1(x)   #1*64*256*256

        x2 = self.Maxpool(x1) #1*64*128*128
        x2 = self.Conv2(x2)   #1*128*128*128

        x3 = self.Maxpool(x2) #1*128*64*64
        x3 = self.Conv3(x3)  #1*256*64*64

        x4 = self.Maxpool(x3) #1*256*32*32
        x4 = self.Conv4(x4) #1*512*32*32


        # decoding + concat path
        d4 = self.Up4(x4) #1*256*64*64
        d4 = torch.cat((x3, d4), dim=1)#1*512*64*64
        d4 = self.Up_conv4(d4) #1*256*64*64

        d3 = self.Up3(d4) #1*128*128*128
        d3 = torch.cat((x2, d3), dim=1) #1*256*128*128
        d3 = self.Up_conv3(d3) #1*128*128*128

        d2 = self.Up2(d3) #1*64*256*256
        d2 = torch.cat((x1, d2), dim=1)#1*128*256*256
        d2 = self.Up_conv2(d2)#1*64*256*256

        d1 = self.Conv_1x1(d2) #1*1*256*256

        return d1




from torch.fft import fftn, fftshift, ifftn, ifftshift


def IFFT_Model(x):
    return ifftshift(ifftn(fftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))

def norm_0_1_Model(arr):
    arr = ((arr - arr.min()) / (arr.max()-arr.min()))
    return arr

def normModel(x: torch.Tensor):
    # group norm
    b, c, h, w = x.shape
    x = x.reshape(b, 2, c // 2 * h * w)
  
    mean = x.mean(dim=2).view(b, 2, 1, 1)
    std = x.std(dim=2).view(b, 2, 1, 1)
   
    x = x.view(b, c, h, w)
    x = (x - mean) / std
  
  
    return x, mean, std



class K_U_Net(nn.Module):
    def __init__(self):
        super(K_U_Net, self).__init__()

        self.Unet_Complex = U_Net_Complex_Real(2)
        

    def forward(self, K_Art):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
        # Imaginary
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
        # Norm 1
        k_corrupt_norm, mu,std  = normModel(K_Art)
        # Complex U-Net
        k_correct_norm = self.Unet_Complex(k_corrupt_norm)
        k_correct_norm1 = k_correct_norm# + k_corrupt_norm
        
        # De norm
        k_correct = k_correct_norm1 * std + mu
        
        # IFFT
        k_correct1 = k_correct[:,0:1,:,:] + 1j * k_correct[:,1:2,:,:]
        real_corrects = []
        for k in k_correct1:
            r = norm_0_1_Model(abs(IFFT_Model(k[0])))
            real_corrects.append(r[None])
        real_corrects1_mag = torch.stack(real_corrects)

        
        return k_correct,  real_corrects1_mag