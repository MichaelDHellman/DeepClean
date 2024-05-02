import torch 
import torch.nn as nn
from torchsummary import summary

#convolutional block
class convBlock(nn.Module):
    def __init__(self, inputZ, outputZ):
        super().__init__()

        self.conv1 = nn.Conv2d(inputZ, outputZ, kernel_size = 4, stride = 1, padding = 2)
        self.bn1 = nn.BatchNorm2d(outputZ)

        self.conv2 = nn.Conv2d(outputZ, outputZ, kernel_size = 4, stride = 1, padding = 2)
        self.bn2 = nn.BatchNorm2d(outputZ)

        self.relu = nn.LeakyReLU(.1)

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
#encoder block
class encoderBlock(nn.Module):
    def __init__(self, inputZ, outputZ):
        super().__init__()
        self.conv = convBlock(inputZ, outputZ)
        self.pool = nn.MaxPool2d((2,2))
    
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
#decoder block
class decoderBlock(nn.Module):
    def __init__(self, inputZ, outputZ):
        super().__init__()
        self.up = nn.ConvTranspose2d(inputZ, outputZ, kernel_size = 2, stride = 1, padding = 0)
        self.conv = convBlock(outputZ + outputZ, outputZ)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x)
        return x
    
#unet architecture
class buildUNET(nn.Module):
    def __init__(self):
        super().__init__()

        """ encoder """
        self.e1 = encoderBlock(3, 64)
        self.e2 = encoderBlock(64, 128)
        self.e3 = encoderBlock(128, 256)
        self.e4 = encoderBlock(256, 512)

        """ bottleneck """
        self.b = convBlock(512, 1024)

        """ decoder """

        self.d1 = decoderBlock(1024, 512)
        self.d2 = decoderBlock(512, 256)
        self.d3 = decoderBlock(256, 128)
        self.d4 = decoderBlock(128, 64)

        """ classifier """
        self.outputs = nn.Conv2d(64, 3, kernel_size = 1, padding = 0)

    def forward(self, inputs):

        """ encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ bottleneck """
        b = self.b(p4)

        """ decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ classifier """
        outputs = self.outputs(d4)

        return outputs
        


if __name__  == "__main__":

    model = buildUNET()

    dummyInput = torch.randn(1, 3, 256, 256)

    output = model(dummyInput)

    print("output shape:", output.shape)


