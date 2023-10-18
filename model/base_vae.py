# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:55:43 2023

@author: tengjunwan
"""


from abc import abstractmethod

import torch.nn as nn

from ._types import List, Callable, Union, Any, TypeVar, Tuple, Tensor


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass 
    
    @abstractmethod
    def loss_function(self, *inputs: List[Tensor], **kwargs):
        pass
    
    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError
        
    def sample(self, batch_size: int, device: int, **kwargs) -> Tensor:
        raise NotImplementedError
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

        