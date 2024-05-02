from dataset.dataset import UIEBDataset
from model.model import ShallowUWNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision import models
import torchvision

# device in use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# instantiate dataset --> copy
trainDataset = UIEBDataset(r"/mnt/e/Workspace/AISMentoring/S24/UIEB/raw",
                          r"/mnt/e/Workspace/AISMentoring/S24/UIEB/reference",
                        force_update=True, train=True)
testDataset = UIEBDataset(r"/mnt/e/Workspace/AISMentoring/S24/UIEB/raw",
                          r"/mnt/e/Workspace/AISMentoring/S24/UIEB/reference",
                        force_update=True, train=False, test=True)

class VGGLoss(nn.Module):
  def __init__(self):
    super(VGGLoss, self).__init__()
    vgg = models.vgg19_bn(pretrained=True)
    self.features = nn.Sequential(*list(vgg.children())[0][:-3]).to(device)
    for param in self.vgg.parameters():
      param.requires_grad = False
  def forward(self, x):
    return self.features(x)

class HybridLoss(nn.Module):
  def __init__(self):
    super(HybridLoss, self).__init__()
    self.vgg_loss = VGGLoss()
    self.mse_loss = nn.MSELoss().to(device)
    self.l1_loss = nn.L1Loss().to(device)

  def forward(self, inp, label):
    vgg_in = self.vgg_loss(inp)
    vgg_label = self.vgg_loss(label)
    mse = self.mse_loss(inp, label)
    l1 = self.l1_loss(vgg_in, vgg_label)
    total = mse + l1
    return total, mse, l1

# instantiate model --> copy
model = ShallowUWNet().to(device)

# create dataloader around UIEBDataset instance --> Batch size = 16
trainDataLoader = DataLoader(trainDataset, batch_size=8, shuffle=True)
testDataLoader = DataLoader(testDataset, batch_size=1, shuffle=False)
# train the shallowUWNet instance --> uses dataloader

# parameters
learn_rate = 0.001
epochs = 100

criterion = HybridLoss()

optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
writer = SummaryWriter()


try:
  for epoch in range(epochs):
    model.train()
    print("Epoch #" + str(epoch) + ":")
    for i, data in enumerate(trainDataLoader):
      inputs = data['raw'].to(device)
      targets = data['reference'].to(device)

      optimizer.zero_grad()
          
      outputs = model(inputs)
      loss, mseloss, l1loss = criterion(outputs, targets)
        
      loss.backward()
      optimizer.step()
      scheduler.step()

      writer.add_scalar('Loss/train', loss.item(), epoch * len(trainDataLoader) + i)

      print(f'MSELoss: {mseloss.item():.4f}' + f'VGGLoss: {l1loss.item():.4f}' + f'total: {loss.item():.4f}')
    
    #test loop
    model.eval()
    with torch.no_grad():
      total_loss = 0
      n = 0
      print("Testing")
      for i, data in enumerate(testDataLoader):
        inputs = data['raw'].to(device)
        targets = data['reference'].to(device)
        
        outputs = model(inputs)
        loss, mseloss, l1loss = criterion(outputs, targets)

        total_loss+=loss.item()
        n = i +1  
        if (i == 50):
          break;

      loss_avg = total_loss / n
      print(f'Test Loss: {loss_avg:.4f}')
    # C:\Users\nicod\PycharmProjects\pythonProject\AIMProject\rawImages
    # get the model loading 
    if (epoch % 1 == 0):
      for i, data in enumerate(testDataLoader):
        inp = data['raw'].to(device)
        target = data['reference'].to(device)
        out = model(inp)
        loss, mseloss, l1loss = criterion(out, target)
        torchvision.utils.save_image(out, "./data/output/" + str(epoch) + "_" + str(i) + ".png", format = "png")
        #p, axarr = plt.subplots(2)
        #axarr[0].imshow(out[0].permute(1, 2, 0).cpu().detach())
        #axarr[1].imshow(inp[0].permute(1, 2, 0).cpu())
        if (i == 10):
          break
      if (epoch % 10 == 0):
        torch.save(model.state_dict(), ('epoch_' + str(epoch) + '.ckpt'))
      

except KeyboardInterrupt:
  print('Training interrupted')
finally:
  writer.close()
  torch.save(model.state_dict(), 'model_checkpoint2.ckpt')
  torch.save(model.state_dict(), 'model_parameters100.pth')


