import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import sys
sys.path.append('../')

from Final.TFCmodel import Architecture
from Final.config import Config
from Final.NTXLoss import NTXentLoss_poly

cfg = Config()

class CNN_Enconder(Architecture):
    time_encoder = nn.Sequential(
        nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=25, stride=3, bias=False, padding=25 // 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35)
        ),

        nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        ),

         nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
    )
    
    frequency_encoder = nn.Sequential(
        nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=25, stride=3, bias=False, padding=(25//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35)
        ),

        nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        ),

        nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
    )

    time_projector = nn.Sequential(
        nn.Linear(640, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    frequency_projector = nn.Sequential(
        nn.Linear(640, 256),
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

    def treat_data(self, X):
        # Xf = torch.reshape(X, (X.size(dim=0), 60, 6))
        X = torch.reshape(X, (X.size(dim=0), 6, 60))
        return X

        
    
