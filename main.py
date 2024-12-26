import os
from data_modules.mosi_module import MosiDataModule
from models.arch import SimpleVLAMGMC
import torch
import numpy as np
# import sacred

# ex = sacred.Experiment('multimodal_research')

from pytorch_lightning import Trainer, seed_everything

AVAIL_GPUS = min(1, torch.cuda.device_count())

def setup_model():
    return SimpleVLAMGMC(name="GMC Test", latent_dim=10, shared_dim=10) 

def setup_data_module(): 
    return MosiDataModule(data_dir="datasets")

def train_model(): 
    model = setup_model()
    data_module = setup_data_module()
    
    trainer = Trainer(devices=AVAIL_GPUS, accelerator='gpu', max_epochs=50, gradient_clip_val=0.8)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    train_model()

    # data_module = setup_data_module()
    # data_module.setup(stage="fit")
    # dataloader = data_module.train_dataloader()
    # print(dataloader.dataset.get_dim(), dataloader.dataset.get_seq_len())
    # for batch in dataloader: 
    #     X, Y = batch
    #     print(np.array(X).shape)
    #     break