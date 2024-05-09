from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    # ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=16):
        super(SE,self).__init__()

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)

        # relu激活
        self.relu = nn.ReLU()

        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, inputs):  # inputs 代表输入特征图
        
        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)

        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)

        x = self.relu(x)

        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)

        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
        # self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            
            nn.ReLU(inplace=True)
        )
    #     self.channel_conv = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
    #         nn.BatchNorm2d(out_channels)
    #     )
    # def forward(self,x):
    #     residual =  x
        
    #     x = self.conv(x)
        
    #     if residual.shape[1] != x.shape[1]:
    #         residual = self.channel_conv(residual)
    #     x = x + residual
    #     return x

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1, #灰度图片1 彩色3
                 num_classes: int = 2, 
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c) #1 64
    
        self.se1 = SE(in_channel=64)
        self.down1 = Down(base_c, base_c * 2) #64 128
        self.se2 = SE(in_channel=128)
        self.down2 = Down(base_c * 2, base_c * 4) #128 256
        self.se3 = SE(in_channel=256)
        self.down3 = Down(base_c * 4, base_c * 8) #256 512
        self.se4 = SE(in_channel=512)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)  #512 512
        self.se5 = SE(in_channel=512)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)  #1024 256
        self.se6 = SE(in_channel=256)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear) #512 128
        self.se7 = SE(in_channel=128)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear) #256 64
        self.se8 = SE(in_channel=64)
        self.up4 = Up(base_c * 2, base_c, bilinear) #128 64
        self.se9 = SE(in_channel=64)
        self.out_conv = OutConv(base_c, num_classes)  

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        x1 = self.in_conv(x) #480*480 64
        x1 = self.se1(x1) 
        
        x2 = self.down1(x1)
        x2 = self.se2(x2)

        x3 = self.down2(x2)
        x3 = self.se3(x3)

        x4 = self.down3(x3)
        x4 = self.se4(x4)

        x5 = self.down4(x4)  # 256
        x5 = self.se4(x5)
        x = self.up1(x5, x4)  # 128
        x = self.se6(x)

        x = self.up2(x, x3)  # 64
        x = self.se7(x)

        x = self.up3(x, x2)  # 32
        x = self.se8(x)

        x = self.up4(x, x1)  # 32
        x = self.se9(x)

        logits = self.out_conv(x)
        print(logits.shape)

        return {"out": logits}
    
    
