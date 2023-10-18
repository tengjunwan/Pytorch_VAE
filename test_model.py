# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:51:09 2023

@author: tengjunwan
"""

import torch

from model import vae_models

model = vae_models["VAE"](3, 128)

x = torch.randn(1, 3, 64, 64)

# test forward
y = model(x)

# test loss_function
loss = model.loss_function(*y)