import torch 
import torch.nn as nn
import numpy as np
import math
from model import model



class buildDiscriminator(nn.Module):

    def __init__(self, filters = 32, kernels = 4, strides = 2, size = 256, channels = 3, leak = 0.2):
        super().__init__()
        #initial vars
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.size = size
        self.channels = channels
        self.leak = leak
        
        def dBlock(filters_in, filters_out):
            layers = [nn.Conv2d(filters_in, filters_out, kernels, stride=strides, padding=1), nn.BatchNorm2d(filters_out,momentum=0.75), nn.LeakyReLU(0.1, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *dBlock(self.channels*2, filters),
            *dBlock(filters, filters*2),
            *dBlock(filters*2, filters*4),
            *dBlock(filters*4, filters*8),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

        
    
    def forward(self, inputs, labels):
        inputcat = torch.cat((inputs, labels), 1)

        out = self.model(inputcat)

        return out

        
if __name__  == "__main__":

    disc = buildDiscriminator()
    print(disc)



    
        

        


