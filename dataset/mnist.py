# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:48:53 2023

@author: tengjunwan
"""

from torchvision.datasets import MNIST
from torchvision import transforms


def get_mnist_dataset(data_dir: str, patch_size: int, **kwargs):
    train_transforms = transforms.Compose([transforms.Resize(patch_size),
                                           transforms.ToTensor(),])
        
    val_transforms = transforms.Compose([transforms.Resize(patch_size),
                                         transforms.ToTensor(),])
    
    train_dataset = MNIST(
        data_dir,
        train=True,
        transform=train_transforms,
        download=False,
        )
    
    val_dataset = MNIST(
        data_dir,
        train=False,
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
    train_dataset, val_dataset = get_mnist_dataset(data_dir, patch_size)