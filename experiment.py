# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:02:28 2023

@author: tengjunwan
"""


import torch.optim as optim
import torch

from model._types import List, Callable, Union, Any, TypeVar, Tuple, Tensor


class VAEExperiment():
    
    def __init__(self, 
                 model, 
                 params: dict):
        self.model = model
        self.params = params
        self.device = None
        self.optimizer = None
        self.scheduler = None
        
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def configure_optimizers(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma = self.params['scheduler_gamma'])

        
    def training_step(self, batch, **kwargs):
        self.model.train()
        real_imgs, labels = batch
        results = self.forward(real_imgs, labels = labels)
        train_loss = self.model.loss_function(
            *results, M_N = self.params['kld_weight'], 
            **kwargs)
        self.optimizer.zero_grad()
        train_loss["loss"].backward()
        self.optimizer.step()

        train_log = {key: val.item() for key, val in train_loss.items()}
        return train_log
    
    def validation_step(self, batch, **kwargs):
        self.model.eval()   
        with torch.no_grad():
            real_imgs, labels = batch
            results = self.forward(real_imgs, labels = labels)
            val_loss = self.model.loss_function(
                *results, **kwargs)
        eval_log = {f"val_{key}": val.item() for key, val \
                    in val_loss.items()}
        
        return eval_log
    
    def on_validation_end(self, batch):
        with torch.no_grad():
            real_imgs, labels = batch
            generate_imgs = self.model.generate(real_imgs)
            sample_imgs = self.model.sample(real_imgs.shape[0], self.device)
            
        return {"generate_imgs": generate_imgs, "sample_imgs": sample_imgs, 
                "real_imgs": real_imgs}
        
    
    def get_current_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        return lr
        