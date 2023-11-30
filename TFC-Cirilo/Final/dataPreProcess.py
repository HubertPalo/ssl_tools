import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('../')

from TFC_Final.ssl_tools.transforms import *
from pathlib import Path

from config import Config
from TFCdataSet import TFCContrastiveDataset

cfg = Config()

class DataPre():
    def __init__(self, dataset, arch):
        self.data_path = Path("../standartized_balanced/" + dataset)
        self.architecture = arch

    def get_data(self, type):
        dataset = pd.read_csv(self.data_path / (type + ".csv"))
        dataset = pd.DataFrame(dataset)

        X = dataset.iloc[:,:360]
        X = torch.tensor(X.values)
        # X = self.treat_data(X)
        
        y = dataset["standard activity code"]
        y = torch.tensor(y.values)

        time_transforms = [
            AddGaussianNoise(std=cfg.jitter_ratio)
        ]

        frequency_transforms = [
            AddRemoveFrequency()
        ]

        train_dataset = TFCContrastiveDataset(
            arquitecture=self.architecture,
            data=X,
            labels=y,
            time_transforms=time_transforms,
            frequency_transforms=frequency_transforms,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=cfg.drop_last,
            num_workers=cfg.num_workers
        )

        return train_loader