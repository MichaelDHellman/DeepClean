import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride = 1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class ShallowUWNet(nn.Module):
    def __init__(self):
        super(ShallowUWNet, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.conv_block1 = ConvBlock(64, 61)
        self.conv_block2 = ConvBlock(64, 61)
        self.conv_block3 = ConvBlock(64, 61)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x_initial = x
        x = self.relu(self.initial_conv(x))

        # Skip connections from the input to each ConvBlock
        x1 = self.conv_block1(x)
        x1_cat = torch.cat((x1, x_initial), dim=1)

        x2 = self.conv_block2(x1_cat)
        x2_cat = torch.cat((x2, x_initial), dim=1)

        x3 = self.conv_block3(x2_cat)
        x3_cat = torch.cat((x3, x_initial), dim=1)

        # Final convolution to get the output image
        x_final = self.final_conv(x3_cat)
        return x_final