from collections import OrderedDict
from models.encoders.SimpleGRUEncoder import SimpleGRUEncoder
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Abstract MultiModal class
class MultiModal(LightningModule):
    def __init__(self, name, num_modalities=2):
        super(MultiModal, self).__init__()

        self.name = name

        self.num_modalities = num_modalities
        self.encoders = nn.ModuleList()
        self.projectors = nn.ModuleList()

        self.joint_processor = nn.Identity()

    def preprocess(self, batch, batch_idx): 
        return None

    def loss(self, prediction, target, batch_representations=None, hyperparams=None, batch_size=None): 
        return None

    def post_process(self, shared_representations): 
        return None
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1, verbose=True),
            "monitor": "val_loss",
            "frequency": 1
        },
        }
        

    def encode(self, x, sample=False):
        # Individual Modality Work
        latent_representations = []
        for id_mod in range(len(x)): 
            if x[id_mod] is not None: 
                encoded_data = self.encoders[id_mod](x[id_mod])
                latent_representations.append(self.projectors[id_mod](encoded_data))
            
        # Joint Fusion Work
        return self.joint_processor(latent_representations)

    def forward(self, x):
        # Forward pass through the modality specific encoders
        batch_representations = []
        shared_representations = []
        for id_mod in range(len(self.encoders) - 1):
            encoded_data = None
            if x[id_mod] is not None: 
                encoded_data = self.encoders[id_mod](x[id_mod])
                shared_representations.append(self.projectors[id_mod](encoded_data))
            batch_representations.append(encoded_data)

        post_processed_output = self.post_process(shared_representations)
        output = self.joint_processor(post_processed_output)

        return output, batch_representations


    def training_step(self, batch, batch_idx):
        X, Y = self.preprocess(batch, batch_idx)

        # Forward pass through the encoders
        output, batch_representations = self.forward(X)

        loss = self.loss(output, Y)
        tqdm_dict = {"loss": loss}

        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output

    def validation_step(self, batch, batch_idx):
        X, Y = self.preprocess(batch, batch_idx)

        # Forward pass through the encoders
        output, batch_representations = self.forward(X)

        loss = self.loss(output, Y)
        tqdm_dict = {"loss": loss}

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return tqdm_dict


# Simple VLAM model implementing GMC 
class SimpleVLAMGMC(MultiModal):
    def __init__(self, name, latent_dim, shared_dim): 
        super(SimpleVLAMGMC, self).__init__(name, num_modalities=3)
            
        self.shared_dim = shared_dim

        # Text
        text_encoder = SimpleGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
        # Vision
        vision_encoder = SimpleGRUEncoder(input_dim=74, hidden_dim=30, latent_dim=latent_dim, timestep=50)
        # Audio 
        audio_encoder = SimpleGRUEncoder(input_dim=35, hidden_dim=30, latent_dim=latent_dim, timestep=50)

        shared_projection_head = nn.Sequential(nn.Linear(latent_dim, shared_dim), nn.ReLU())

        self.encoders = nn.ModuleList([text_encoder, vision_encoder, audio_encoder])
        self.projectors = nn.ModuleList([shared_projection_head, shared_projection_head, shared_projection_head])
        self.joint_processor = nn.Linear(self.shared_dim, 1)

    def preprocess(self, batch, batch_idx): 
        batch_X, batch_Y, batch_META = batch[0], batch[1], batch[2]
        sample_ind, text, audio, vision = batch_X
        target_data = batch_Y.squeeze(-1)
        return [text, audio, vision], target_data

    def loss(self, prediction, target, batch_representations=None, hyperparams=None, batch_size=None):
        criterion = nn.L1Loss()
        loss = torch.mean(criterion(prediction, target))
        return loss

    def post_process(self, shared_representations):
        average_result = torch.mean(torch.stack(shared_representations), dim=0)
        return average_result