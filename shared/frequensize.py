import torch.nn as nn
import torch
import torch.nn.functional as F
import torch

fft2 = torch.fft.fft2
ifft2 = torch.fft.ifft2
fftshift = torch.fft.fftshift
ifftshift = torch.fft.ifftshift

def K2Img(arr):
    arr = ifftshift(ifft2(fftshift(arr))) 
    return arr


def Img2K(arr):
    arr = fftshift(fft2(ifftshift(arr)))
    return arr


s = 105
def FREQ_OLD(Imgall, USE_CUDA= False):
    Los = []; Highs = []
    
    for Img in Imgall:
        Img = Img[None]
        LowFreqFilter = torch.ones((256, 256), dtype=torch.uint8) * 1  # White background
        LowFreqFilter[:s, :] = 0  # Top border
        LowFreqFilter[-s:, :] = 0  # Bottom border
        LowFreqFilter[:, :s] = 0  # Left border
        LowFreqFilter[:, -s:] = 0  # Right border


        HighFreqFilter = 1- LowFreqFilter
        if(USE_CUDA):
            LowFreqFilter = LowFreqFilter.to("cuda")
            HighFreqFilter = HighFreqFilter.to("cuda")
    
        # fft take +ve values -> norm  then norm back
        m = 0
        if(Img.min()<0):
            m = Img.min()
            Img = Img + -m

        K = Img2K(Img)
        LowImg  = (abs(K2Img(K*LowFreqFilter))+m).type(torch.float32)   
        
        HighImg = (abs(K2Img(K*HighFreqFilter))+m).type(torch.float32)   

        Los.append(LowImg)
        Highs.append(HighImg)
    return K,torch.concat(Los),torch.concat(Highs), LowFreqFilter, HighFreqFilter




def De_FREQ_OLD(Lowall, highall, USE_CUDA= False):
    IMGs = []
    
    for LOW,HIGH in zip(Lowall,highall):
        LOW = LOW[None]
        HIGH = HIGH[None]
        LowFreqFilter = torch.ones((256, 256), dtype=torch.uint8) * 1  # White background
        LowFreqFilter[:s, :] = 0  # Top border
        LowFreqFilter[-s:, :] = 0  # Bottom border
        LowFreqFilter[:, :s] = 0  # Left border
        LowFreqFilter[:, -s:] = 0  # Right border
   
        HighFreqFilter = 1- LowFreqFilter
        if(USE_CUDA):
            LowFreqFilter = LowFreqFilter.to("cuda")
            HighFreqFilter = HighFreqFilter.to("cuda")
    
        # fft take +ve values -> norm  then norm back
        print(LOW.shape)
        m = 0
        if(LOW.min()<0):
            m = LOW.min()
            LOW = LOW + -m
            HIGH = HIGH + -m

        K = Img2K(LOW)*HighFreqFilter #+ Img2K(HIGH)*HighFreqFilter
        
        Img = (abs(K2Img(K))+m).type(torch.float32)   

        IMGs.append(Img)

    return torch.concat(IMGs)











# def FREQ(Imgall, USE_CUDA= False):
#     Los = []; Highs = []
    
#     for Img in Imgall:
#         Img = Img[None]
#         s = 100
#         LowFreqFilter = torch.ones((256, 256), dtype=torch.uint8) * 1  # White background
#         LowFreqFilter[:s, :] = 0  # Top border
#         LowFreqFilter[-s:, :] = 0  # Bottom border
#         LowFreqFilter[:, :s] = 0  # Left border
#         LowFreqFilter[:, -s:] = 0  # Right border
        
#         if(USE_CUDA):
#             LowFreqFilter = LowFreqFilter.to("cuda")
    
#         # fft take +ve values -> norm  then norm back
#         m = 0
#         if(Img.min()<0):
#             m = Img.min()
#             Img = Img + -m

#         K = Img2K(Img)
#         LowImg  = (abs(K2Img(K*LowFreqFilter))+m).type(torch.float32)   
        
#         Highkernel = torch.FloatTensor([[-1, -2, 1], [-2, 0, 2], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        
#         if(USE_CUDA):
#             Highkernel = Highkernel.cuda()
#         HighImg = (F.conv2d(Img+m, Highkernel, stride=1, padding=1))

#         Los.append(LowImg)
#         Highs.append(HighImg)

#     return torch.concat(Los)[0,0], torch.concat(Highs)[0,0]

def FREQ(Imgall, USE_CUDA= False):
    Los = []; Highs = []
    Highkernel = torch.FloatTensor([[-1, -2, 1], [-2, 0, 2], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
    if(USE_CUDA):
        Highkernel = Highkernel.cuda()
    
    for Img in Imgall:
        Img = Img[None]
        
        HighImg = (F.conv2d(Img, Highkernel, stride=1, padding=1))
        Highs.append(HighImg)

    return torch.concat(Highs)








beta = 2
def FREQ_Downsample_Minus(XHR, USE_CUDA= False):
    XLR = F.avg_pool2d(XHR, beta)

    # Step 2: Upsample
    XLF = F.interpolate(XLR, scale_factor=beta, mode='bilinear', align_corners=False)
    # Step 3: Generate high-frequency image
    XHF = XHR - XLF


    return XLF, XHF












import numpy as np
import pywt
import albumentations as A
aug = A.Resize(256, 256, p=1)

def FREQ_WaveLet(Imgall):
    Los = []; Highs = []
    try:
        Imgall = Imgall.cpu().detach()
    except:
        pass
    for Img in Imgall:
        image = Img[0]
        LL, (LH, HL, HH) = pywt.dwt2(image, "db2")
        LowImg = aug(image=LL)['image']
        mix = LH+ HL+ HH
        HighImg = aug(image=mix)['image']

        Los.append(torch.tensor(LowImg)[None])
        Highs.append(torch.tensor(HighImg)[None])
    return torch.stack(Los), torch.stack(Highs)




















import math
def gaussian_filter(rows,columns,sigma):
    if rows%2==0:
        center_x = rows/2
    else:
        center_x = rows/2 + 1
    if columns%2==0:
        center_y = columns/2
    else:
        center_y = columns/2 + 1
 
    def gaussian(i,j):
        value = math.exp(-1.0 * ((i - center_x)**2 + (j - center_y)**2) / (2 * sigma**2))
        return value
 
    gaussian_array = torch.zeros((rows,columns))

    for i in range(0,rows):
        for j in range(0,columns):
            gaussian_array[i][j] = gaussian(i,j)
    
    return gaussian_array


def FREQ_guass(Imgall, USE_CUDA= False):
    Los = []; Highs = []
    
    for Img in Imgall.cuda():
        image = Img[0]
        n,m = image.shape
        gaussian_matrixLow = gaussian_filter(n,m,25).cuda()
        gaussian_matrixHigh = (1 - gaussian_filter(n,m,25)).cuda()

        dft = torch.fft.fft2(image)
        dftshift = torch.fft.fftshift(dft)
        
        filterImageLow = dftshift * gaussian_matrixLow
        ifftshiftLow = torch.fft.ifftshift(filterImageLow)
        ifftImageLow = torch.fft.ifft2(ifftshiftLow)

        filterImageHigh = dftshift * gaussian_matrixHigh
        ifftshiftHigh = torch.fft.ifftshift(filterImageHigh)
        ifftImageHigh = torch.fft.ifft2(ifftshiftHigh)
        Los.append(torch.real(ifftImageLow)[None][None])
        Highs.append( torch.real(ifftImageHigh)[None][None])
    if(USE_CUDA):
        return torch.concat(Los).cuda(), torch.concat(Highs).cuda()
    return torch.concat(Los), torch.concat(Highs)


















def get_wav_two(x,in_channels=1, pool=True,USE_CUDA=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 3))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 3))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=3, stride=1, padding=1, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=3, stride=1, padding=1, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=3, stride=1, padding=1, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=3, stride=1, padding=1, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    LL = LL.cuda().float()
    LH = LH.cuda().float()
    HL = HL.cuda().float()
    HH = HH.cuda().float()
    x = x.cuda()
    return LL(x), LH(x)+HL(x)+ HH(x)
