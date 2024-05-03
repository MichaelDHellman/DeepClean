import torch
import torchvision
from model.gen import buildUNET
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib import pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import UIEBDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    model = buildUNET().to(device)
    model.load_state_dict(torch.load("./checkpoints/cGANGen.ckpt"))
    model.eval()
    trainDataset = UIEBDataset(r"/mnt/e/Workspace/AISMentoring/S24/UIEB/raw",
                          r"/mnt/e/Workspace/AISMentoring/S24/UIEB/reference",
                        force_update=True, train=True, doAugment=False)
    trainDataLoader = DataLoader(trainDataset, batch_size=1, shuffle=True)
    for i, sample in enumerate(trainDataLoader):
        print("workign")
        out = model(sample["raw"].to(device))
        torchvision.utils.save_image(out, "./data/output/" + str(i) + "_" + str(i) + ".png", format = "png")


