
import torch
import torchvision.transforms as T
import os
import pandas as pd 
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from skimage import io, transform



class RandomTransforms(object):#randomly transforms the raw and reference images in the sample dictionary
    def __init__(self, scale_range=(0.8, 1.2), output_size=(256, 256)):#initializes the RandomTransforms object with the given scale range and output size
        self.output_size = output_size
        self.transforms = T.Compose([
            T.RandomResizedCrop(output_size, scale=scale_range),
            T.Lambda(lambda img: TF.rotate(img, random.choice([0, 90, 180 ,270]))),  # Rotate by 0, 90, 180, 270 degrees via lambda function
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])
        
    def __call__(self, sample):#transforms the raw and reference images in the sample dictionary
        raw, reference = sample['raw'], sample['reference']
        
        # Apply transformations to the raw and reference images in the sample dictionary  
        raw_transformed = self.transforms(raw.unsqueeze(0))
        reference_transformed = self.transforms(reference.unsqueeze(0))
        

        
        sample['raw'] = raw_transformed.squeeze(0)
        sample['reference'] = reference_transformed.squeeze(0)
        return sample#returns the sample dictionary with the transformed raw and reference images

class UIEBDataset(Dataset):
    #3/8/2024 - UIEB is switched to add the data onto csv files and stores the path to raw and reference directories
    #basically the same as the previous dataset class but now it reads from csv files
    def __init__(self, raw_dir, reference_dir, dims =(256, 256), doAugment = True, train = True, test = False, force_update = False):
        self.raw_dir = raw_dir
        self.reference_dir = reference_dir
        self.dims = dims# uses given dims and passes it to the Rescale object
        self.rescale = Rescale(dims)
        self.tensorfy = ToTensor()
        self.doAugment = True
        self.doAugment = doAugment#now a boolean flag to determine if augmentation should be performed(used to be a function)
        self.random_transforms = RandomTransforms(scale_range = (0.8, 1.2),output_size = dims) if doAugment else None
        if not (os.path.isfile(os.path.join(os.getcwd(), r"train_list.csv"))) or force_update:
            self.build_test_list(raw_dir, reference_dir)
        if train:
            self.samples = pd.read_csv(os.path.join(os.getcwd(), r"train_list.csv"))
        elif test:
            self.samples = pd.read_csv(os.path.join(os.getcwd(), r"test.csv"))
        else:
            pass
        print(self.samples.shape)
        print(self.samples)


    # invokes the augment function to apply random transformations to the raw and reference images in the sample dictionary
    def augment(self, sample):
        print('error finder') # Check if augmentation should be performed
            # Apply random transformations to both raw and reference images
        sample = self.random_transforms(sample)
        return sample
    def build_test_list(self, raw_dir, reference_dir):#builds the test list csv file from the raw and reference directories
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


    def __len__(self):#returns the number of samples in the dataset(returns the number of row in the csv file)
        return self.samples.shape[0]
    
    def standardize(self, sample):#standardizes the raw and reference images in the sample dictionary
        sample = self.rescale(sample)
        sample = self.tensorfy(sample)#tensorfy is a ToTensor object that converts the raw and reference images to tensors
        return sample
    
    def __getitem__(self, idx):#returns the sample at the given index in the dataset (returns the row at the given index in the csv file)
        if torch.is_tensor(idx):
            idx = idx.tolist()#batching the data

        raw_image = io.imread(self.samples.iloc[idx, 0])#idx is a number between 0 and the number of samples in the dataset
        reference_image = io.imread(self.samples.iloc[idx, 1])#reference image is the second column in the csv file, changes based on the idx
        sample = {'raw': raw_image, 'reference': reference_image}#returns a dictionary with the raw and reference images
        
        sample = self.standardize(sample)#standardizing the sample dictionary

        #if self.doAugment:#if the augment function is not None, the sample is augmented
        sample = self.augment(sample)
        
        return sample
        

class Rescale(object):#rescales the image to the given dimensions 

    def __init__(self, out_dims):#initializes the Rescale object with the given dimensions
        self.out_dims = out_dims

    def __call__(self, sample):#rescales the raw and reference images to the given dimensions
        raw, reference = sample['raw'], sample['reference']
        h, w = raw.shape[:2]
        new_h, new_w = self.out_dims

        raw = transform.resize(raw, (new_h, new_w))#resizes the raw image to the given dimensions
        reference = transform.resize(reference, (new_h, new_w))#resizes the reference image to the given dimensions

        return {'raw': raw, 'reference': reference}#returns a dictionary with the rescaled raw and reference images

class ToTensor(object):#converts the raw and reference images to tensors

    def __call__(self, sample):
        raw, reference = sample['raw'], sample['reference']

        raw, reference = raw.transpose((2,0,1)), reference.transpose((2,0,1))#changes the format of the raw and reference images to match the format of the pytorch tensors format
        return {'raw': torch.from_numpy(raw), 'reference': torch.from_numpy(reference)}#returns a dictionary with the raw and reference images as pytoch tensors format
    
    
if __name__ == "__main__":
    dataset = UIEBDataset(r"/mnt/c/users/neezu/Desktop/Class/UTD/AIM/UIEB/raw-890", r"/mnt/c/users/neezu/Desktop/Class/UTD/AIM/UIEB/reference-890", force_update= "False")
    for i, sample in enumerate(dataset):#iterates through the dataset and displays the raw and reference images for each sample
        print("Displaying sample pair " + str(i))

        p, axarr = plt.subplots(2)#creates a figure with 2 subplots 

        axarr[0].imshow(sample['raw'].permute(1, 2, 0))#displays the raw image in the first subplot we change it to match matplotlib's format
        axarr[1].imshow(sample['reference'].permute(1, 2, 0))#displays the reference image in the second subplot we change it to match matplotlib's format

        plt.show()#displays the figure with the raw and reference images