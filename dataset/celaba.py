# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:50:30 2023

@author: tengjunwan
"""

from torchvision.datasets import CelebA
from torchvision import transforms



def get_celeba_dataset(data_dir: str, patch_size: int, **kwargs):
    
    
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.CenterCrop(148),
                                           transforms.Resize(patch_size),
                                           transforms.ToTensor(),])
        
    val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.CenterCrop(148),
                                         transforms.Resize(patch_size),
                                         transforms.ToTensor(),])
    
    train_dataset = CelebA(
        data_dir,
        split='train',
        transform=train_transforms,
        download=False,
        )
    
    val_dataset = CelebA(
        data_dir,
        split='test',
        transform=val_transforms,
        download=False,
        )
    
    return train_dataset, val_dataset
    


if __name__== "__main__":
    # test MNIST
    data_dir = "D:/learnspace/dataset"
    patch_size = 256
    # dataset = MNIST(data_dir, download=False)
    # test CelebA
    train_dataset, val_dataset = get_celeba_dataset(data_dir, patch_size)
    
    