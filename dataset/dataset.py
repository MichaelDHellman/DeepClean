import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib import pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt


class Rescale(object):

    def __init__(self, out_dims):
        self.out_dims = out_dims

    def __call__(self, sample):
        raw, reference = sample['raw'], sample['reference']
        h, w = raw.shape[:2]
        new_h, new_w = self.out_dims

        raw = transform.resize(raw, (new_h, new_w))
        reference = transform.resize(reference, (new_h, new_w))

        return {'raw': raw, 'reference': reference}

class ToTensor(object):

    def __call__(self, sample):
        raw, reference = sample['raw'], sample['reference']

        raw, reference = raw.transpose((2,0,1)), reference.transpose((2,0,1))
        return {'raw': torch.from_numpy(raw).type(torch.FloatTensor), 'reference': torch.from_numpy(reference).type(torch.FloatTensor)}

class UIEBDataset(Dataset):
    def __init__(self, raw_dir, reference_dir, dims =(256, 256), doAugment = True, train = True, test = False, force_update = False):
        self.raw_dir = raw_dir
        self.reference_dir = reference_dir
        self.doAugment = True
        self.dims = dims
        self.rescale = Rescale(dims)
        self.randomErase = RandomErasing()
        self.doAugment = doAugment
        self.dims = dims
        self.rescale = Rescale(dims)
        self.tensorfy = ToTensor()
        if not (os.path.isfile(os.path.join(os.getcwd(), r"train_list.csv"))) or force_update:
            self.build_test_list(raw_dir, reference_dir)
        if train:
            self.samples = pd.read_csv(os.path.join(os.getcwd(), r"train_list.csv"))
        elif test:
            self.samples = pd.read_csv(os.path.join(os.getcwd(), r"test_list.csv"))
        else:
            pass
        print(self.samples.shape)
        print(self.samples)

    def build_test_list(self, raw_dir, reference_dir):
        f_list = [(os.fsdecode(os.path.join(raw_dir, i)) + "," + (os.fsdecode(os.path.join(reference_dir, i)))) for i in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, i))]
        train_size = int(len(f_list)*.8)
        train_str = "raw, reference\n" + "\n".join(f_list[0:train_size])
        test_str = "raw, reference\n" + "\n".join(f_list[train_size:])
        train = open(os.path.join(os.getcwd(), r"train_list.csv"), "w")
        train.write(train_str)
        train.close()
        test = open(os.path.join(os.getcwd(), r"test_list.csv"), "w")
        test.write(test_str)
        test.close()

    def __len__(self):
        return self.samples.shape[0]
    
    def augment(self, sample):
        self.randomErase(sample)
        return sample

    def __getitem__(self, idx):

        raw_image = io.imread(self.samples.iloc[idx, 0])
        reference_image = io.imread(self.samples.iloc[idx, 1])

        sample = {'raw': raw_image, 'reference': reference_image}

        sample = self.standardize(sample)

        if self.doAugment:
            sample = self.augment(sample)

        return sample

    def standardize(self, sample):
        sample = self.rescale(sample)
        sample = self.tensorfy(sample)
        return sample
    

class RandomErasing(object):
    def __init__(self, probability=0.7, scale=(0.04, 0.06), ratio=(0.3, 3)):
        self.probability = probability
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        transform = transforms.RandomErasing(p=self.probability, scale=self.scale, ratio=self.ratio)
        state = torch.get_rng_state()
        sample['raw'] = transform(sample['raw'])
        torch.set_rng_state(state)
        sample['reference'] = transform(sample['reference'])
        return sample





if __name__ == "__main__":
    dataset = UIEBDataset(r"/mnt/e/Projects/Datasets/UIEB/raw-890", r"/mnt/e/Projects/Datasets/UIEB/reference-890", force_update=False)

    for i, sample in enumerate(dataset):
        print("Displaying sample pair " + str(i))

        p, axarr = plt.subplots(2)

        axarr[0].imshow(sample['raw'].permute(1, 2, 0))
        axarr[1].imshow(sample['reference'].permute(1, 2, 0))

        plt.show()