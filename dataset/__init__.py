# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:49:59 2023

@author: tengjunwan
"""

from .celaba import get_celeba_dataset
from .mnist import get_mnist_dataset



vae_dataset_func = {"celeba": get_celeba_dataset,
                    "mnist": get_mnist_dataset}


def get_dataset(dataset_name: str, **kwargs):
    train_dataset, val_dataset = vae_dataset_func[dataset_name](**kwargs)
    return train_dataset, val_dataset
    