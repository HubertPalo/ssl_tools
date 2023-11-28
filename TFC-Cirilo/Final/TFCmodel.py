import torch
import pytorch_lightning as pl
import torch.nn as nn
from typing import Any

import sys
sys.path.append('../../')
from TFC_Final.ssl_tools.transforms import *

class Architecture():
    def __init__(
        self, 
        time_encoder, 
        time_projector, 
        frequency_econder, 
        frequency_projector,
        nxtent_criterion, 
    ):
        self.time_encoder = time_encoder
        self.time_projector = time_projector
        self.frequency_encoder = frequency_econder
        self.frequency_projector = frequency_projector
        self.nxtent_criterion = nxtent_criterion

class TFC(pl.LightningModule):
    lr: float = 1e-3
    loss_lambda: float = 0.2

    def __init__(
        self,
        architecture: Architecture,
    ):
        super().__init__()
        self.save_hyperparameters() 
        self.architecture = architecture
        self.time_encoder = architecture.time_encoder
        self.time_projector = architecture.time_projector
        self.frequency_encoder = architecture.frequency_encoder
        self.frequency_projector = architecture.frequency_projector
        self.nxtent_criterion = architecture.nxtent_criterion

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.time_encoder(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.time_projector(h_time)

        """Frequency-based contrastive encoder"""
        f = self.frequency_encoder(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.frequency_projector(h_freq)

        final_embedding = torch.concat(z_time, z_freq)

        return h_time, z_time, h_freq, z_freq, final_embedding

    def configure_optimizers(self) -> Any:
        learnable_parameters = (
            list(self.time_encoder.parameters()) +
            list(self.time_projector.parameters()) +
            list(self.frequency_encoder.parameters()) +
            list(self.frequency_projector.parameters())
        )
        optimizer = torch.optim.Adam(learnable_parameters, lr= self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, labels, aug1, data_f, aug1_f = batch
        
        """Producing embeddings"""
        h_t, z_t, h_f, z_f = self.forward(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1, aug1_f)

        """Calculate losses"""
        loss_time = self.nxtent_criterion(h_t, h_t_aug)
        loss_freq = self.nxtent_criterion(h_f, h_f_aug)
        loss_consistency = self.nxtent_criterion(z_t, z_f)
        loss = (self.loss_lambda * (loss_time + loss_freq)) + loss_consistency
        
        # log loss, only to appear on epoch
        self.log('time_loss', loss_time, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('freq_loss', loss_freq, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('consistency_loss', loss_consistency, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

