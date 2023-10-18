# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:52:56 2023

@author: tengjunwan
"""

from .vanilla_vae import VanillaVAE


vae_models = {"VAE": VanillaVAE,
              "VanillaVAE": VanillaVAE,
              }


def get_model(model_name, **kwargs):
    model = vae_models[model_name](**kwargs)
    
    return model




