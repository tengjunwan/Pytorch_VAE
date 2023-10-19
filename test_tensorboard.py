# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:05:09 2023

@author: tengjunwan
"""

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from model import get_model

writer = SummaryWriter()
# add scalar
for i in range(100):
    writer.add_scalar("a", np.random.random(), i)
    writer.add_scalar("b", np.random.random(), i)
    writer.add_scalar("c", np.random.random(), i)
    writer.add_scalar("d", np.random.random(), i)
    
# add scalars
r = 5
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)

# add hitogram
for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)

# add a image
for i in range(5):
    im = Image.open("demo.jpg")
    im_np = np.array(im).astype(np.float32)
    im_np_norm = im_np / 255.0
    writer.add_image("songjinyagnzi", im_np_norm, global_step=i,
                      dataformats="HWC")


transforms = transforms.Compose([transforms.Resize(64),
                                 transforms.ToTensor(),])
dataset = MNIST("D:/aigc/dataset", download=False, transform=transforms)
dataloader = DataLoader(dataset, batch_size=32, num_workers=0, )
batch = next(iter(dataloader))
ims = batch[0]

# add images
for i in range(5):
    writer.add_image("mnist", ims, global_step=i,
                      dataformats="NCHW")
    
# model graph
model = get_model("VanillaVAE", in_channels=1, latent_dim=128)
writer.add_graph(model, ims)

writer.close()