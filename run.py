# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:49:04 2023

@author: tengjunwan
"""

import yaml
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import get_model
from experiment import VAEExperiment
from trainer import Trainer


def main():
    pass


# load config
config_path = "./config/vae.yaml"
with open(config_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# load data
train_dataset, val_dataset = get_dataset(**config["data_params"])
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["data_params"]["train_batch_size"],                  
    num_workers=config["data_params"]["num_workers"],
    shuffle=True,
    pin_memory=config['trainer_params']['device'] != 'cpu'
    )

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["data_params"]["val_batch_size"],                  
    num_workers=config["data_params"]["num_workers"],
    shuffle=False,
    pin_memory=config['trainer_params']['device'] != 'cpu'
    )

# load model
model = get_model(**config["model_params"])

# load experiment
exp = VAEExperiment(model, config["exp_params"])

# l

# trainer
trainer = Trainer(**config["trainer_params"])

trainer.fit(exp, train_dataloader, val_dataloader)

if __name__ == "__main__":
    pass