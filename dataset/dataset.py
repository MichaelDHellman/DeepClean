import os
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import pandas as pd
from skimage import io, transform
from torchvision import transforms


class ImageBlur:
    def __init__(self, kernel_size, sigma=(1.0, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.blurrr = transforms.GaussianBlur(kernel_size, sigma)

    def __call__(self, image):
        if not torch.is_tensor(image):
            print("Image has to be a tensor")
            return
        if image.dim() not in [3, 4]:
            print("Image has not the right dimensions")
        return self.blurrr(image)


class UIEBDataset(Dataset):

    def __init__(self, raw_dir, reference_dir, dims=(256, 256), augment=None, train=True, test=False,
                 force_update=False):
        self.raw_dir = raw_dir
        self.reference_dir = reference_dir
        self.augment = None
        self.dims = dims
        self.rescale = Rescale(dims)
        self.tensorfy = ToTensor()
        self.image_blur = ImageBlur((9, 9), sigma=(1.0, 2.0))

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

    def build_test_list(self, raw_dir, reference_dir):
        f_list = [(os.fsdecode(os.path.join(raw_dir, i)) + "," + (os.fsdecode(os.path.join(reference_dir, i)))) for i in
                  os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, i))]
        train_size = int(len(f_list) * .8)
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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        raw_image = io.imread(self.samples.iloc[idx, 0])
        reference_image = io.imread(self.samples.iloc[idx, 1])

        sample = {'raw': raw_image, 'reference': reference_image}
        sample = self.standardize(sample)
        if self.augment:
            sample = self.augment(sample)

        return sample

    def standardize(self, sample):
        sample = self.rescale(sample)
        sample = self.tensorfy(sample)

        sample['raw'] = self.image_blur(sample['raw'])
        sample['reference'] = self.image_blur(sample['reference'])
        return sample


class ToTensor(object):

    def __call__(self, sample):
        raw, reference = sample['raw'], sample['reference']

        raw, reference = raw.transpose((2, 0, 1)), reference.transpose((2, 0, 1))
        return {'raw': torch.from_numpy(raw), 'reference': torch.from_numpy(reference)}


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


if __name__ == "__main__":
    dataset = UIEBDataset(r"/mnt/c/users/nicod/PycharmProjects/pythonProject/AIMProject/rawImages/",
                          r"/mnt/c/users/nicod/PycharmProjects/pythonProject/AIMProject/referenceImages/",
                          force_update=True)

    for i, sample in enumerate(dataset):
        print('Displaying sample pair', str(i))
        p, axarr = plt.subplots(2)
        axarr[0].imshow(sample['raw'].permute(1, 2, 0))
        axarr[1].imshow(sample['reference'].permute(1, 2, 0))

        plt.show()
