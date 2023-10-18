# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:25:33 2023

@author: tengjunwan
"""

from pathlib import Path

from tqdm import tqdm
import torch


class Trainer():
    
    def __init__(self, max_epochs, device, checkpoint_dir, log_dir):
        self.max_epochs = max_epochs
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
    
    def fit(self, experiment, train_loader, val_loader):
        experiment.model.to(self.device)
        experiment.configure_optimizers()
        
        
        for epoch in range(self.max_epochs):
            current_lr = experiment.get_current_lr()
            train_progress = tqdm(
                train_loader, 
                desc=f"Train:{epoch + 1}/{self.max_epochs}|lr:{current_lr:.3f}")
            # train
            running_tr_loss = []
            for xs, labels in tqdm(train_loader):
                xs, labels = self._to_device(xs, labels)
                train_log = experiment.training_step((xs, labels))
                running_tr_loss.append(train_log["loss"])
                # print
                train_msg = [f"{k}:{v:.2f}" for k, v in train_log.items()]
                train_progress.set_postfix_str("|".join(train_msg))
                
            # val
            val_progress = tqdm(val_loader, 
                                desc=f"Epoch {epoch + 1}/Training")
            running_val_loss = []
            for xs, labels in val_progress:
                xs, labels = self._to_device(xs, labels)
                val_log = experiment.validation_step((xs, labels))
                running_val_loss.append(val_log["val_loss"])
                # print
                val_msg = [f"{k}:{v:.2f}" for k, v in val_log.items()]
                val_progress.set_postfix_str("|".join(val_msg))
                
            experiment.scheduler.step()
            
            average_tr_loss = sum(running_tr_loss) / len(running_tr_loss)
            average_val_loss = sum(running_val_loss) / len(running_val_loss)
            self._save_checkpoint(experiment.model,
                f"{epoch}_{average_tr_loss:.2f}_{average_val_loss:.2f}.pth")
            
    
    def _to_device(self, *args):
        ret = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.to(self.device)
                ret.append(arg)
                
        return ret
    
    def _save_checkpoint(self, model, save_name):
        if not self.checkpoint_dir.is_dir():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.checkpoint_dir / save_name
        torch.save(model.state_dict(), str(save_path))
        
    def _log():
        pass
    
    