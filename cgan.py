import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn
import wandb
from model.gen import buildUNET
from model.discrim import buildDiscriminator
from dataset.dataset import UIEBDataset
from torchvision import models
from tqdm.autonotebook import tqdm, trange


IMAGE_DIMS = (256, 256)
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = (1, IMAGE_DIMS[0] // 16, IMAGE_DIMS[1] // 16)

class VGGLayers(nn.Module):
  def __init__(self):
    super(VGGLayers, self).__init__()
    self.vgg = models.vgg19_bn(pretrained=True)
    self.features = nn.Sequential(*list(self.vgg.children())[0][:-3]).to(device)
    for param in self.vgg.parameters():
      param.requires_grad = False
  def forward(self, x):
    return self.features(x)
  
class VGGLoss(nn.Module):
  def __init__(self):
    super(VGGLoss, self).__init__()
    self.vgg = VGGLayers()

  def forward(self, x, y):
    vgg_x = self.vgg(x)
    vgg_y = self.vgg(y)
    return torch.mean((vgg_x - vgg_y)**2)


def trainWrapper():
    run = wandb.init(

        project = "cGAN-Deepclean",
        config = {
            "learning_rate": 1e-3,
            "epochs": 100,
        },
    )

    generator = buildUNET().to(device)
    discriminator = buildDiscriminator().to(device)

    criterion = nn.MSELoss().to(device)
    vgg = VGGLoss().to(device)
    L1 = nn.L1Loss().to(device)

    generator_optim = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    trainDataset = UIEBDataset(r"/mnt/e/Workspace/AISMentoring/S24/UIEB/raw",
                          r"/mnt/e/Workspace/AISMentoring/S24/UIEB/reference",
                        force_update=True, train=True)
    testDataset = UIEBDataset(r"/mnt/e/Workspace/AISMentoring/S24/UIEB/raw",
                            r"/mnt/e/Workspace/AISMentoring/S24/UIEB/reference",
                            force_update=True, train=False, test=True)
    
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle = True)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle = False)

    writer = SummaryWriter(f"data/logs")

    discriminator.train()
    generator.train()
    mseLosslst = []
    l1Losslst = []
    vggLosslst = []
    totalLosslst = []
    for epoch in range(epochs):
      print("Epoch #" + str(epoch) + ":")
      mseLosstmp = 0
      l1Losstmp = 0
      vggLosstmp = 0
      totalLosstmp = 0
      for i, sample in enumerate(trainLoader):
        raw = sample["raw"].to(device)
        gt = sample["reference"].to(device)

        rLabel = torch.ones((raw.size(0), *PATCH_SIZE), dtype = raw.dtype).to(device)
        fLabel = torch.zeros((raw.size(0), *PATCH_SIZE), dtype = raw.dtype).to(device)
        discriminator_optim.zero_grad()

        fake = generator(raw)
        rPred = discriminator(gt, raw)
        rLoss = criterion(rPred, rLabel)
        fPred = discriminator(fake, raw)
        fLoss = criterion(fPred, fLabel)
        tLoss = (rLoss + fLoss)
        tLoss.backward()
        discriminator_optim.step()

        generator_optim.zero_grad()
        fake = generator(raw)
        fPred = discriminator(fake, raw)
        mseLoss = criterion(fPred, rLabel)
        l1Loss = L1(fake, gt)
        vggLoss = vgg(fake, gt)

        totalLoss = mseLoss + (vggLoss * 7) + (l1Loss * 7)
        mseLosstmp += mseLoss.item()
        vggLosstmp += vggLoss.item()
        l1Losstmp += l1Loss.item()
        totalLosstmp += totalLoss.item()
        print(f"Loss: {totalLoss.item()}:.4f")
        totalLoss.backward()
        generator_optim.step()
      mseLosslst.append(mseLosstmp/len(trainLoader))
      vggLosslst.append(vggLosstmp/len(trainLoader))
      l1Losslst.append(l1Losstmp/len(trainLoader))
      totalLosslst.append(totalLosstmp/len(trainLoader))
      wandb.log({"G_mse": mseLosslst[epoch], "G_vgg": vggLosslst[epoch], "G_l1": l1Losslst[epoch], "G_total": totalLosslst[epoch]}, commit = True)

      if (epoch % 1 == 0):
        for i, data in enumerate(trainLoader):
          inp = data['raw'].to(device)
          target = data['reference'].to(device)
          out = generator(inp)
          torchvision.utils.save_image(out, "./output/" + str(epoch) + "_" + str(i) + ".png", format = "png")
          if (i == 10):
            break
        if ((epoch + 1) % 2 == 0):
          torch.save(discriminator.state_dict(), ('./checkpoints/D_epoch_' + str(epoch) + '.ckpt'))
          torch.save(generator.state_dict(), ('./checkpoints/G_epoch_' + str(epoch) + '.ckpt'))

if __name__ == "__main__":
   trainWrapper()