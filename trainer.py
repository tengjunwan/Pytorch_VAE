# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:25:33 2023

@author: tengjunwan
"""

from pathlib import Path

from tqdm import tqdm
import torch 
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    
    def __init__(self, max_epochs, device, checkpoint_dir, log_dir):
        self.max_epochs = max_epochs
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
    
    def fit(self, experiment, train_loader, val_loader):
        experiment.model.to(self.device)
        experiment.configure_optimizers()
        experiment.device = self.device
        tb = SummaryWriter(str(self.log_dir / experiment.params["exp_name"]))
        
        for epoch in range(self.max_epochs):
            current_lr = experiment.get_current_lr()
            train_progress = tqdm(enumerate(train_loader), 
                                  total=len(train_loader))
            train_msg = \
                f"Train:{epoch + 1}/{self.max_epochs}|lr:{current_lr:.3f}|"
            # train
            running_tr = {}
            for i, (xs, labels) in train_progress:
                xs, labels = self._to_device(xs, labels)
                train_log = experiment.training_step((xs, labels))
                
                for k, v in train_log.items():
                    running_tr.setdefault(k, []).append(v)
                
                if i % 49 == 0:
                    for k, v in train_log.items():
                        tb.add_scalar(k, v, i + epoch * len(train_loader))
                
                running_train_msg = "|".join(
                    f"{k}:{v:.3f}" for k, v in train_log.items())
                train_progress.set_description(train_msg + running_train_msg)

            # val
            val_progress = tqdm(val_loader, total=len(val_loader),
                                desc=f"Val:{epoch + 1}/{self.max_epochs}")
            running_val = {}
            for xs, labels in val_progress:
                xs, labels = self._to_device(xs, labels)
                val_log = experiment.validation_step((xs, labels))
                for k, v in val_log.items():
                    running_val.setdefault(k, []).append(v)
                    
            # get average
            for k in running_tr.keys():
                running_tr[k] = sum(running_tr[k]) / len(running_tr[k])
            for k in running_val.keys():
                running_val[k] = sum(running_val[k]) / len(running_val[k])
                
            val_msg = "|".join(f"{k}:{v:.3f}" for k, v in running_val.items())
            print("val result:" + val_msg)
                
                
            experiment.scheduler.step()
            
            for k, v in running_val.items():
                tb.add_scalar(k, v, epoch)
                
            # visualize
            xs, labels = next(iter(val_loader))
            xs, labels = self._to_device(xs, labels)
            vis_result = experiment.on_validation_end((xs, labels))
            for k, v in vis_result.items():
                tb.add_images(k, v, global_step=epoch)
            
            self._save_checkpoint(experiment.model,
                f"epoch: {epoch}_{running_tr['loss']:.3f}"+
                f"_{running_val['val_loss']:.3f}.pth")
            
            tb.close()
            
    
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
        
    
    