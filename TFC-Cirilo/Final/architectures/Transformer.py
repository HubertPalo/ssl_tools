import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import sys
sys.path.append('../')

from Final.TFCmodel import Architecture
from Final.config import Config
from Final.NTXLoss import NTXentLoss_poly

cfg = Config()

class Transformer_Enconder(Architecture):
    time_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            60, dim_feedforward=2 * cfg.length_alignment, nhead=2
        ),
        num_layers=2,
    )
    frequency_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            60, dim_feedforward=2 * cfg.length_alignment, nhead=2
        ),
        num_layers=2,
    )

    time_projector = nn.Sequential(
        nn.Linear(cfg.length_alignment, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    frequency_projector = nn.Sequential(
        nn.Linear(cfg.length_alignment, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    nxtent = NTXentLoss_poly(
        batch_size=cfg.batch_size,
        temperature=cfg.temperature,
        use_cosine_similarity=cfg.use_cosine_similarity,
    )

    def __init__(self):
        super().__init__(
            self.time_encoder, 
            self.time_projector, 
            self.frequency_encoder, 
            self.frequency_projector, 
            self.nxtent
        )

    def get_data_time(self, X):
        Xf = torch.reshape(X, (X.size(dim=0), 60, 6))
        X = torch.reshape(X, (X.size(dim=0), 6, 60))

        for i, x in enumerate(X):
            x = torch.stack((x[0],x[1],x[2],x[3],x[4],x[5]), dim = 1)
            Xf[i] = x
        
        Xf = Xf.permute(0, 2, 1)
        return Xf

    def get_data_freq(self, X):
        Xf = torch.reshape(X, (X.size(dim=0), 60, 6))
        X = torch.reshape(X, (X.size(dim=0), 6, 60))

        for i, x in enumerate(X):
            x = torch.stack((x[0],x[1],x[2],x[3],x[4],x[5]), dim = 1)
            Xf[i] = x
        
        Xf = Xf.permute(0, 2, 1)
        return Xf

        
    
