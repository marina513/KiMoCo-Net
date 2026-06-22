import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


	
# CBAM --------------------------------------------	
# Convolutional Block Attention Module(CBAM) block    


class CBAMBlock(nn.Module):
	def __init__(self, in_channels,Input_Height, ratio=8):
		super(CBAMBlock, self).__init__()
		self.channel_attention = ChannelAttention(in_channels,Input_Height, ratio)
		self.spatial_attention = SpatialAttention()

	def forward(self, x):
		x = self.channel_attention(x)
		x = self.spatial_attention(x)
		return x


class ChannelAttention(nn.Module):
	def __init__(self, in_channels,Input_Height, ratio=8):
		super(ChannelAttention, self).__init__()
		
		self.shared_layer_one = nn.Linear(in_channels, in_channels // ratio)
		self.shared_layer_two = nn.Linear(in_channels // ratio, in_channels)
		
		self.avg = nn.AvgPool2d(kernel_size=Input_Height)
		self.max = nn.MaxPool2d(kernel_size=Input_Height)
		
		self.Relu = nn.ReLU()
		self.sig = nn.Sigmoid()

	def forward(self, x):
		# Average pooling
		avg_pool = self.avg(x)
		avg_pool = avg_pool.reshape(avg_pool.shape[0],1,1,avg_pool.shape[1])  # 1 * 1  * 1 * 64
		avg_pool = self.shared_layer_one(avg_pool)
		avg_pool = self.Relu(avg_pool)
		avg_pool = self.shared_layer_two(avg_pool)
		avg_pool = self.sig(avg_pool)
		avg_pool = avg_pool.reshape(avg_pool.shape[0],avg_pool.shape[-1],1,1)
		# Max pooling
		max_pool = self.max(x)
		max_pool = max_pool.reshape(max_pool.shape[0],1,1,max_pool.shape[1])  # 1 * 1  * 1 * 64
		max_pool = self.shared_layer_one(max_pool)
		max_pool = self.Relu(max_pool)
		max_pool = self.shared_layer_two(max_pool)

		max_pool = self.sig(max_pool)
		max_pool = max_pool.reshape(max_pool.shape[0],max_pool.shape[-1],1,1)

		# Channel attention
		cbam_feature = avg_pool + max_pool
		return x * cbam_feature
		
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        cbam_feature = torch.sigmoid(self.conv(concat))
        return x * cbam_feature
	

    






























	
class UNet_pt(nn.Module):
	def __init__(self,input_channels):
		super(UNet_pt, self).__init__()
		
		# Block 1 in Contracting Path
		self.conv_L1_1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm1_1 = nn.BatchNorm2d(32)
		self.relu1_1 = nn.ReLU()
		
		self.conv_L1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm1_2 = nn.BatchNorm2d(32)
		self.relu1_2 = nn.ReLU()

		self.Avgpool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.cbam1_2 = CBAMBlock(32,256)
		
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 2 in Contracting Path
		self.conv_L2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm2_1 = nn.BatchNorm2d(64)
		self.relu2_1 = nn.ReLU()
		self.drop2 = nn.Dropout(0.2)
		self.conv_L2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm2_2 = nn.BatchNorm2d(64)
		self.relu2_2 = nn.ReLU()

		self.Avgpool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.cbam2_2 = CBAMBlock(64,128)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 3 in Contracting Path
		self.conv_L3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm3_1 = nn.BatchNorm2d(128)
		self.relu3_1 = nn.ReLU()
		
		self.conv_L3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm3_2 = nn.BatchNorm2d(128)
		self.relu3_2 = nn.ReLU()

		self.Avgpool3_2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.cbam3_2 = CBAMBlock(128,64)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 4 in Contracting Path
		self.conv_L4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm4_1 = nn.BatchNorm2d(256)
		self.relu4_1 = nn.ReLU()
		
		self.conv_L4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm4_2 = nn.BatchNorm2d(256)
		self.relu4_2 = nn.ReLU()

		self.cbam4_2 = CBAMBlock(256,32)
	
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 1 in Expansive Path
		self.upsample1 =  nn.Upsample(scale_factor=2, mode="bilinear")
		self.conv_up1_1 = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)
		self.bn_up1_1 = nn.BatchNorm2d(128)
		self.relu_up1_1 = nn.ReLU()

		self.conv_up1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.bn_up1_2 = nn.BatchNorm2d(128)
		self.relu_up1_2 = nn.ReLU()

		self.cbamup_1 = CBAMBlock(128,64)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 2 in Expansive Path
		self.upsample2 =  nn.Upsample(scale_factor=2, mode="bilinear")
		self.conv_up2_1 = nn.Conv2d(128+64, 64, kernel_size=3, padding=1)
		self.bn_up2_1 = nn.BatchNorm2d(64)
		self.relu_up2_1 = nn.ReLU()

		self.conv_up2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.bn_up2_2 = nn.BatchNorm2d(64)
		self.relu_up2_2 = nn.ReLU()

		self.cbamup_2 = CBAMBlock(64,128)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 3 in Expansive Path
		self.upsample3 =  nn.Upsample(scale_factor=2, mode="bilinear")
		self.conv_up3_1 = nn.Conv2d(64+32, 32, kernel_size=3, padding=1)
		self.bn_up3_1 = nn.BatchNorm2d(32)
		self.relu_up3_1 = nn.ReLU()

		self.conv_up3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
		self.bn_up3_2 = nn.BatchNorm2d(32)
		self.relu_up3_2 = nn.ReLU()

		self.cbamup_3 = CBAMBlock(32,256)

		self.conv_Final = nn.Conv2d(32, 1, kernel_size=3, padding=1)


	def forward(self, x):
		
		# Block 1 in Contracting Path
		conv1 = self.conv_L1_1(x)         
		conv1 = self.batch_norm1_1(conv1)   
		conv1 = self.relu1_1(conv1)         

		conv1 = self.conv_L1_2(conv1)         
		conv1 = self.batch_norm1_2(conv1)   
		conv1 = self.relu1_2(conv1)           

		conv1 = self.cbam1_2(conv1)         
		o = self.Avgpool1_2(conv1) # 1*32*128*128

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 2 in Contracting Path
		conv2 = self.conv_L2_1(o)         
		conv2 = self.batch_norm2_1(conv2)   
		conv2 = self.relu2_1(conv2)         
		conv2 = self.drop2(conv2)
		conv2 = self.conv_L2_2(conv2)         
		conv2 = self.batch_norm2_2(conv2)   
		conv2 = self.relu2_2(conv2)           

		conv2 = self.cbam2_2(conv2)         
		o = self.Avgpool2_2(conv2) # 1*64*64*64

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 3 in Contracting Path
		conv3 = self.conv_L3_1(o)         
		conv3 = self.batch_norm3_1(conv3)   
		conv3 = self.relu3_1(conv3)         
		conv3 = self.conv_L3_2(conv3)         
		conv3 = self.batch_norm3_2(conv3)   
		conv3 = self.relu3_2(conv3)           

		conv3 = self.cbam3_2(conv3)         
		o = self.Avgpool3_2(conv3) # 1*128*32*32

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Transition
		conv4 = self.conv_L4_1(o)         
		conv4 = self.batch_norm4_1(conv4)   
		conv4 = self.relu4_1(conv4)         
		conv4 = self.conv_L4_2(conv4)         
		conv4 = self.batch_norm4_2(conv4)   
		conv4 = self.relu4_2(conv4)           

		conv4 = self.cbam4_2(conv4)   # 1*256*32*32    

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 1 in Expansive Path
		up1 = self.upsample1(conv4) #1*256*64*64
		up1 = torch.concat((up1,conv3),1)
		deconv1 = self.conv_up1_1(up1) 
		deconv1 = self.bn_up1_1(deconv1)
		deconv1 = self.relu_up1_1(deconv1)
		
		deconv1 = self.conv_up1_2(deconv1)
		deconv1 = self.bn_up1_2(deconv1)
		deconv1 = self.relu_up1_2(deconv1)
		deconv1 = self.cbamup_1(deconv1)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 2 in Expansive Path
		up2 = self.upsample1(deconv1) #1*128*128*128
		up2 = torch.concat((up2,conv2),1)
		deconv2 = self.conv_up2_1(up2) 
		deconv2 = self.bn_up2_1(deconv2)
		deconv2 = self.relu_up2_1(deconv2)
		
		deconv2 = self.conv_up2_2(deconv2) 
		deconv2 = self.bn_up2_2(deconv2)
		deconv2 = self.relu_up2_2(deconv2)
		deconv2 = self.cbamup_2(deconv2) 

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# Block 2 in Expansive Path
		up3 = self.upsample2(deconv2) #1*64*256*256
		up3 = torch.concat((up3,conv1),1)
		deconv3 = self.conv_up3_1(up3) 
		deconv3 = self.bn_up3_1(deconv3)
		deconv3 = self.relu_up3_1(deconv3)
		
		deconv3 = self.conv_up3_2(deconv3)
		deconv3 = self.bn_up3_2(deconv3)
		deconv3 = self.relu_up3_2(deconv3)
		deconv3 = self.cbamup_3(deconv3) 

		output = self.conv_Final(deconv3)

		return output


class Correction_Multi_input_pt(nn.Module):
	def __init__(self):
		super(Correction_Multi_input_pt, self).__init__()
		
		# Feature extracion
		self.conv_L1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm1 = nn.BatchNorm2d(32)
		self.relu1 = nn.ReLU()

		self.conv_L2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm2 = nn.BatchNorm2d(32)
		self.relu2 = nn.ReLU()

		self.conv_L3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1,dilation=1)  
		self.batch_norm3 = nn.BatchNorm2d(32)
		self.relu3 = nn.ReLU()

		self.unet1 = UNet_pt(32*3)
		self.unet2 = UNet_pt(32*3+1)

	def forward(self, x, bef, aft):
		
		# Feature extracion
		conv1 = self.conv_L1(x)         
		conv1 = self.batch_norm1(conv1)   
		conv1 = self.relu1(conv1)           

		conv2 = self.conv_L2(bef)         
		conv2 = self.batch_norm2(conv2)   
		conv2 = self.relu2(conv2)           

		conv3 = self.conv_L3(aft)         
		conv3 = self.batch_norm3(conv3)   
		conv3 = self.relu3(conv3)     

		input_concat = torch.concat((conv1,conv2,conv3),1) 
		
		## Two Stacked Nets:
		pred_1  = self.unet1(input_concat)
		input_2 = torch.concat([input_concat, pred_1], 1) 
		pred_2  = self.unet2(input_2) # 

		return pred_2

