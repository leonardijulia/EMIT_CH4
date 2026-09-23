import torch
import torch.nn as nn
import numpy as np

# UNet components

# Since at each level of the UNet, we have two convolutional layers, we can define a double convolution block as a separate class. 
class DoubleConv(nn.Module):
    """3x3 Conv => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# The upsampling block in the UNet architecture consists of an upsampling operation followed by a double convolution.
class UpsampleBlock(nn.Module):
    """Up-conv from UNet architecture: Bilinear upsampling followed by a double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.up(x)
    

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = UpsampleBlock(512, 256)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = UpsampleBlock(256, 128)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = UpsampleBlock(128, 64)
        self.dec1 = DoubleConv(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(nn.MaxPool2d(kernel_size=2)(enc1_out))
        enc3_out = self.enc3(nn.MaxPool2d(kernel_size=2)(enc2_out))

        # Bottleneck
        bottleneck_out = self.bottleneck(nn.MaxPool2d(kernel_size=2)(enc3_out))

        # Decoder        
        up3_out = self.up3(bottleneck_out)
        dec3_out = self.dec3(torch.cat((up3_out, enc3_out), dim=1))
        
        up2_out = self.up2(dec3_out)
        dec2_out = self.dec2(torch.cat((up2_out, enc2_out), dim=1))
        
        up1_out = self.up1(dec2_out)
        dec1_out = self.dec1(torch.cat((up1_out, enc1_out), dim=1))
        
        logits = self.final_conv(dec1_out)
        return logits