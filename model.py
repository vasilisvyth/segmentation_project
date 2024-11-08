import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Double_conv_batch_relu(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Double_conv_batch_relu,self).__init__()
        self.w = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.w(x)
        return x

class Up_conv_batch_relu(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Up_conv_batch_relu,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Conv_batch_relu(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Conv_batch_relu,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.conv_batch_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.conv_batch_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.conv_batch_sigmoid = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.conv_batch_g(g)
        x1 = self.conv_batch_x(x)
        psi = self.relu(g1+x1)
        psi = self.conv_batch_sigmoid(psi)

        return x*psi


class UNet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(UNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = Double_conv_batch_relu(ch_in=img_ch,ch_out=64)
        self.Conv2 = Double_conv_batch_relu(ch_in=64,ch_out=128)
        self.Conv3 = Double_conv_batch_relu(ch_in=128,ch_out=256)
        self.Conv4 = Double_conv_batch_relu(ch_in=256,ch_out=512)
        self.Conv5 = Double_conv_batch_relu(ch_in=512,ch_out=1024)

        self.Up5 = Up_conv_batch_relu(ch_in=1024,ch_out=512)
        self.Up_conv5 = Double_conv_batch_relu(ch_in=1024, ch_out=512)

        self.Up4 = Up_conv_batch_relu(ch_in=512,ch_out=256)
        self.Up_conv4 = Double_conv_batch_relu(ch_in=512, ch_out=256)
        
        self.Up3 = Up_conv_batch_relu(ch_in=256,ch_out=128)
        self.Up_conv3 = Double_conv_batch_relu(ch_in=256, ch_out=128)
        
        self.Up2 = Up_conv_batch_relu(ch_in=128,ch_out=64)
        self.Up_conv2 = Double_conv_batch_relu(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttentionUNet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(AttentionUNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = Double_conv_batch_relu(ch_in=img_ch,ch_out=64)
        self.Conv2 = Double_conv_batch_relu(ch_in=64,ch_out=128)
        self.Conv3 = Double_conv_batch_relu(ch_in=128,ch_out=256)
        self.Conv4 = Double_conv_batch_relu(ch_in=256,ch_out=512)
        self.Conv5 = Double_conv_batch_relu(ch_in=512,ch_out=1024)

        self.Up5 = Up_conv_batch_relu(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = Double_conv_batch_relu(ch_in=1024, ch_out=512)

        self.Up4 = Up_conv_batch_relu(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = Double_conv_batch_relu(ch_in=512, ch_out=256)
        
        self.Up3 = Up_conv_batch_relu(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = Double_conv_batch_relu(ch_in=256, ch_out=128)
        
        self.Up2 = Up_conv_batch_relu(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = Double_conv_batch_relu(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1