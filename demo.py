import torch
import torchvision
from model.model import ShallowUWNet
from torch.utils.data import DataLoader
import torch.nn as nn

if __name__ == "__main__":
    model = ShallowUWNet()
    model.load_state_dict(torch.load("/mnt/e/Workspace/AISMentoring/S24/deepclean/epoch_1000.ckpt"))
    model.eval()
