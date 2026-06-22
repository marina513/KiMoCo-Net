from scipy import ndimage
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torcheval.metrics import PeakSignalNoiseRatio
from shared.ssim import SSIM
ssim_calc = SSIM()


def plot_Img(Img, slicee=10,title="", rotate=False,angle=90):
    try:
        Img = Img.cpu().detach()
    except:
        pass
    if(str(type(Img)) != "<class 'torch.Tensor'>"):
        Img = torch.tensor(Img)
    Img = Img.squeeze().squeeze()

    from shared.CreateRoffetBlur import CreteRoffetBlur
    b  = CreteRoffetBlur(2)
    
    from shared.gradient_metrics import aes
    val,canny,grad = aes(Img)

    title = ("BLUR: "+str(float(b(Img.squeeze()[None][None])))[:5] ) +" aes:" +  str(val)[:7]

    print(title)
    if(rotate):
        Img = ndimage.rotate(Img, angle)
    plt.axis('off')
    plt.title(title)
    plt.imshow( Img ,cmap='gray')


from skimage.metrics import  structural_similarity
def ssim_SK(gt: np.ndarray, pred: np.ndarray, maxval = None):
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max()# - gt.min() if maxval is None else maxval
    print(maxval)

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval)

    return ssim / gt.shape[0]

def plot_2_Imgs(Img1, Img2, title="",title1 = "", title2="", save_path = "", rotate= False,angle=90):

    Img1 = Img1.squeeze()
    Img2 = Img2.squeeze()

    try:
        Img1 = Img1.cpu().detach()
        Img2 = Img2.cpu().detach()
    except:
        pass
    if(str(type(Img1)) != "<class 'torch.Tensor'>"):
        Img1 = torch.tensor(Img1)
    if(str(type(Img2)) != "<class 'torch.Tensor'>"):
        Img2 = torch.tensor(Img2)


    #SSIM = ssim_SK(np.array(Img1[None]), np.array(Img2[None]))[0]
    SSIM = float(ssim_calc(Img1[None][None],Img2[None][None]))


    metric = PeakSignalNoiseRatio()
    metric.update(Img1,Img2)
   # print("PSNR: ",float(metric.compute()))
    from shared.CreateRoffetBlur import CreteRoffetBlur
    b  = CreteRoffetBlur(2)
    print("BLUR: ",b(Img2.squeeze()[None][None]))


    if(rotate):
        Img1 = ndimage.rotate(Img1, angle)
        Img2 = ndimage.rotate(Img2, angle)
        
    fig = plt.figure()
    fig.suptitle(title + str(SSIM)[:5],y=0.85)
    ax1 = fig.add_subplot(121)
    #ax1.set_xticklabels([])
    #ax1.set_yticklabels([])
    ax2 = fig.add_subplot(122)
    #ax2.set_xticklabels([])
    #ax2.set_yticklabels([])
    ax1.title.set_text(title1)
    ax2.title.set_text(title2)
    ax1.imshow(Img1 ,cmap='gray')
    ax2.imshow(Img2 ,cmap='gray')

    ax1.axis('off')
    ax2.axis('off')
    
    plt.show()
    if(len(save_path)>1):
        fig.savefig(save_path)








def norm_0_1(arr):
    arr = ((arr - arr.min()) / (arr.max()-arr.min()))
    return arr








def plot_3_Imgs(Img1, Img2, Img3,
                title="", title1="", title2="", title3="",save_path="", rotate=False,angle=90):
    Img1 = Img1.squeeze()
    Img2 = Img2.squeeze()
    Img3 = Img3.squeeze()

    try:
        Img1 = Img1.cpu().detach()
        Img2 = Img2.cpu().detach()
        Img3 = Img3.cpu().detach()
    except:
        pass
    if(str(type(Img1)) != "<class 'torch.Tensor'>"):
        Img1 = torch.tensor(Img1)
    if(str(type(Img2)) != "<class 'torch.Tensor'>"):
        Img2 = torch.tensor(Img2)
    if(str(type(Img3)) != "<class 'torch.Tensor'>"):
        Img3 = torch.tensor(Img3)

    Img1 = (Img1).clone().detach()
    Img2 = (Img2).clone().detach()
    Img3 = (Img3).clone().detach()

    metric = PeakSignalNoiseRatio()
    metric.update(Img1,Img2)
    PSNR1 = float(metric.compute())

    #print("PSNR: ",PSNR1)

    
    metric = PeakSignalNoiseRatio()
    metric.update(Img1,Img3)
    PSNR2 = float(metric.compute())



    #SSIM1 = ssim_SK(np.array(Img1[None]), np.array(Img2[None]))#(Img1, Img2)
    #SSIM2 = ssim_SK(np.array(Img1[None]), np.array(Img3[None]))#calc_loss(Img1, Img3)
    SSIM1 = float(ssim_calc(Img1[None][None],Img2[None][None]))
    SSIM2 = float(ssim_calc(Img1[None][None],Img3[None][None]))


    if(rotate):
        Img1 = ndimage.rotate(Img1, angle)
        Img2 = ndimage.rotate(Img2, angle)
        Img3 = ndimage.rotate(Img3, angle)
    fig = plt.figure()
    fig.suptitle(title, y=0.75)# + str(SSIM1)[:5] + " & " + str(SSIM2)[:5], y=0.75)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #ax4 = fig.add_subplot(144)
    ax1.title.set_text(title1 )
    ax2.title.set_text(title2+ str(SSIM1)[:6]+ " / " + str(PSNR1)[:6])
    ax3.title.set_text(title3+ str(SSIM2)[:6] + " / " + str(PSNR2)[:6])
    #ax4.title.set_text("Diff")
    ax1.imshow(Img1 ,cmap='gray')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax2.imshow(Img2 ,cmap='gray')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    ax3.imshow(Img3 ,cmap='gray')
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.show()
    if(len(save_path)>1):
        fig.savefig(save_path)


from shared.eagle_loss import GradientVariance
GV = GradientVariance(3)

def plot_4_Imgs(Img1, Img2, Img3, Img4,
                title="", title1="", title2="", title3="",title4="",
                save_path="", rotate=False):

    try:
        Img1 = Img1.cpu().detach()
        Img2 = Img2.cpu().detach()
        Img3 = Img3.cpu().detach()
        Img4 = Img4.cpu().detach()
    except:
        pass
    if(str(type(Img1)) != "<class 'torch.Tensor'>"):
        Img1 = torch.tensor(Img1)
    if(str(type(Img2)) != "<class 'torch.Tensor'>"):
        Img2 = torch.tensor(Img2)
    if(str(type(Img3)) != "<class 'torch.Tensor'>"):
        Img3 = torch.tensor(Img3)
    if(str(type(Img4)) != "<class 'torch.Tensor'>"):
        Img4 = torch.tensor(Img4)

    Img1 = Img1.squeeze().float()
    Img2 = Img2.squeeze().float()
    Img3 = Img3.squeeze().float()
    Img4 = Img4.squeeze().float()

    metric = PeakSignalNoiseRatio()
    metric.update(Img1,Img2)
    print("PSNR: ",float(metric.compute()))

    
    metric = PeakSignalNoiseRatio()
    metric.update(Img1,Img3)
    print("PSNR: ",float(metric.compute()))

    
    metric = PeakSignalNoiseRatio()
    metric.update(Img1,Img4)
    print("PSNR: ",float(metric.compute()))


    #SSIM1 = ssim_SK(np.array(Img1[None]), np.array(Img2[None]))#calc_loss(Img1, Img2)
    #SSIM2 = ssim_SK(np.array(Img1[None]), np.array(Img3[None]))#calc_loss(Img1, Img3)
   #SSIM3 = ssim_SK(np.array(Img1[None]), np.array(Img4[None]))#calc_loss(Img1, Img4)
    SSIM1 = float(ssim_calc(Img1[None][None],Img2[None][None]))
    SSIM2 = float(ssim_calc(Img1[None][None],Img3[None][None]))
    SSIM3 = float(ssim_calc(Img1[None][None],Img4[None][None]))

    from skimage import color, data, measure

 #   print("Blur:",measure.blur_effect(((Img1-Img1.min()))/((Img1.max()-Img1.min())).cpu().detach(),h_size=11))
 #   print("Blur:",measure.blur_effect(((Img2-Img2.min()))/((Img2.max()-Img2.min())).cpu().detach(),h_size=11))
  #  print("Blur:",measure.blur_effect(((Img3-Img3.min()))/((Img3.max()-Img3.min())).cpu().detach(),h_size=11))
   # print("Blur:",measure.blur_effect(((Img4-Img4.min()))/((Img4.max()-Img4.min())).cpu().detach(),h_size=11))


    #print("GV:",GV(Img1[None][None],Img2[None][None]))
    #print("GV:",GV(Img1[None][None],Img3[None][None]))
    #print("GV:",GV(Img1[None][None],Img4[None][None]))

    if(rotate):
        Img1 = ndimage.rotate(Img1, 90)
        Img2 = ndimage.rotate(Img2, 90)
        Img3 = ndimage.rotate(Img3, 90)
        Img4 = ndimage.rotate(Img4, 90)


    fig = plt.figure()
    fig.suptitle(title, y=0.75)# + str(SSIM1)[:5] + " & " + str(SSIM2)[:5], y=0.75)
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    ax1.title.set_text(title1)
    ax2.title.set_text(title2+ str(SSIM1)[1:5])
    ax3.title.set_text(title3+ str(SSIM2)[1:5])
    ax4.title.set_text(title4+ str(SSIM3)[1:5])
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax1.imshow(Img1 ,cmap='gray')
    ax2.imshow(Img2 ,cmap='gray')
    ax3.imshow(Img3 ,cmap='gray')
    ax4.imshow(Img4 ,cmap='gray')
    plt.show()
    if(len(save_path)>1):
        fig.savefig(save_path)


