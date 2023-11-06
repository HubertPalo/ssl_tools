#!/usr/bin/env python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help = "Dataset utilizado para treino")
parser.add_argument('--clsf', type=str, help = "Classificador utilizado para treino")

 
args = parser.parse_args()

dataset = args.dataset
clsf = args.clsf

print("Ola", clsf)

import sys

from sklearn.exceptions import UndefinedMetricWarning
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
    
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import Union

from ssl_tools.transforms import *

from typing import Any
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# Run parameters
num_epochs = 200
accelerator = "gpu"
num_gpus = 1
strategy = "ddp"
checkpoint_dir = Path("save_models")
tfc_pretrain_checkpoint = checkpoint_dir / "preTrain/"
tfc_classifier_checkpoint = checkpoint_dir / "fineTune/" 
result_dir = Path("results/")
result_dir.mkdir(exist_ok=True, parents=True)
results_tfc_classifier = result_dir / ("results:" + dataset + "-" + clsf + ".yaml")
batch_size = 60

data_path_train = Path("standartized_balanced/" + dataset)
data_path_test = Path("standartized_balanced/" + dataset)

# NXTent Loss
jitter_ratio = 2
length_alignment = 360
drop_last = True
num_workers = 10
learning_rate = 3e-4
temperature = 0.2
use_cosine_similarity = True
is_subset = False
n_classes = 7

class TFCContrastiveDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor = None,
        length_alignment: int = 360,
        time_transforms: Union[Transform, List[Transform]] = None,
        frequency_transforms: Union[Transform, List[Transform]] = None,
    ):
        assert len(data) == len(labels), "Data and labels must have the same length"
        
        self.data_time = data
        self.labels = labels
        self.length_alignment = length_alignment
        self.time_transforms = time_transforms or []
        self.frequency_transforms = frequency_transforms or []
        
        if not isinstance(self.time_transforms, list):
            self.time_transforms = [self.time_transforms]
        if not isinstance(self.frequency_transforms, list):
            self.frequency_transforms = [self.frequency_transforms]

        if len(self.data_time.shape) < 3:
            self.data_time = self.data_time.unsqueeze(2)

        if self.data_time.shape.index(min(self.data_time.shape)) != 1:
            self.data_time = self.data_time.permute(0, 2, 1)

        """Align the data to the same length, removing the extra features"""
        self.data_time = self.data_time[:, :1, : self.length_alignment]
        
        """Calculcate the FFT of the data and apply the transforms (if any)"""
        self.data_freq = torch.fft.fft(self.data_time).abs()
        
        # This could be done in the __getitem__ method
        # For now, we do it here to be more similar to the original implementation
        self.data_time_augmented = self.apply_transforms(self.data_time, self.time_transforms)
        self.data_freq_augmented = self.apply_transforms(self.data_freq, self.frequency_transforms)
        
    def apply_transforms(self, x: torch.Tensor, transforms: List[Transform]) -> torch.Tensor:
        for transform in transforms:
            x = transform.fit_transform(x)
        return x
        
    def __len__(self):
        return len(self.data_time)
    
    def __getitem__(self, index):
        # Time processing
        return (
            self.data_time[index].float(),
            self.labels[index],
            self.data_time_augmented[index].float(),
            self.data_freq[index].float(),
            self.data_freq_augmented[index].float(),
        )


class TFC(nn.Module):
    def __init__(
        self,
        time_encoder: nn.Module,
        frequency_encoder: nn.Module,
        time_projector: nn.Module,
        frequency_projector: nn.Module,
    ):
        super().__init__()

        self.time_encoder = time_encoder
        self.time_projector = time_projector
        self.frequency_encoder = frequency_encoder
        self.frequency_projector = frequency_projector

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

        return h_time, z_time, h_freq, z_freq

    def out_data(self, batch):
        data, labels, aug1, data_f, aug1_f = batch

        """Producing embeddings"""
        h_t, z_t, h_f, z_f = self(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self(aug1, aug1_f)

        """Add supervised loss"""
        fea_concat = torch.cat((z_t, z_f), dim=1)
        return fea_concat


from typing import Any

class TFC_classifier(pl.LightningModule):
    def __init__(
        self,
        tfc_model: torch.nn.Module,
        classifier: torch.nn.Module,
        n_classes: int = 2,
    ):
        super().__init__()
        self.tfc_model = tfc_model
        self.classifier = classifier
        self.n_classes = n_classes

    def train(self, data):
        X=[]
        y=[]
        for batch in data:
            d, labels, aug1, data_f, aug1_f = batch
            out = self.tfc_model.out_data(batch)
            for i in out:
                X.append(i)
            for i in labels:
                y.append(i)

        self.classifier.fit(X, y)
    
    def predict(self, data):
        X=[]
        y=[]
        for batch in data:
            d, labels, aug1, data_f, aug1_f = batch
            out = self.tfc_model.out_data(batch)
            for i in out:
                X.append(i)
            for i in labels:
                y.append(i)

        y = np.array(y)
        pred = self.classifier.predict(X)
        
        print(len(pred), len(y))

        correct = 0
        for i in range(len(pred)):
            if pred[i] == y[i]:
                correct += 1

        return float(correct / len(pred))


def build_model():
    time_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            length_alignment, dim_feedforward=2 * length_alignment, nhead=2
        ),
        num_layers=2,
    )
    frequency_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            length_alignment, dim_feedforward=2 * length_alignment, nhead=2
        ),
        num_layers=2,
    )

    time_projector = nn.Sequential(
        nn.Linear(length_alignment, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    frequency_projector = nn.Sequential(
        nn.Linear(length_alignment, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    tfc_model = TFC(
        time_encoder=time_encoder,
        frequency_encoder=frequency_encoder,
        time_projector=time_projector,
        frequency_projector=frequency_projector,
    )
    
    tfc_model.load_state_dict(torch.load(tfc_pretrain_checkpoint / ("PreTrain:" + dataset + ".ckpt"))["state_dict"])
    
    # Freezing
    for param in tfc_model.parameters():
        param.requires_grad = False

    if clsf == 'RF':
        classifier = RandomForestClassifier(max_depth=10, random_state=0)
    elif clsf == 'SVM':
        classifier = svm.SVC(decision_function_shape='ovo')
    elif clsf == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors = 100)

    tfc_classifier = TFC_classifier(
        tfc_model=tfc_model,
        classifier=classifier,
        n_classes=n_classes,
    )
    
    return tfc_classifier


def get_data():
    dataset_train = pd.read_csv(data_path_train / "train.csv")
    dataset_train = pd.DataFrame(dataset_train)
    dataset_validation = pd.read_csv(data_path_train / "validation.csv")
    dataset_validation = pd.DataFrame(dataset_validation)
    dataset_test = pd.read_csv(data_path_test / "test.csv")
    dataset_test = pd.DataFrame(dataset_test)

    X_train = dataset_train.iloc[:,:360]
    y_train = dataset_train["standard activity code"]
    X_validation = dataset_validation.iloc[:,:360]
    y_validation = dataset_validation["standard activity code"]
    X_test = dataset_test.iloc[:,:360]
    y_test = dataset_test["standard activity code"]

    X_train = torch.tensor(X_train.values)
    y_train = torch.tensor(y_train.values)
    X_validation = torch.tensor(X_validation.values)
    y_validation = torch.tensor(y_validation.values)
    X_test = torch.tensor(X_test.values)
    y_test = torch.tensor(y_test.values)

    train_dataset = TFCContrastiveDataset(
        data=X_train,
        labels=y_train,
        time_transforms=None,
        frequency_transforms=None,
    )

    validation_dataset = TFCContrastiveDataset(
        data=X_validation,
        labels=y_validation,
        time_transforms=None,
        frequency_transforms=None,
    )

    test_dataset = TFCContrastiveDataset(
        data=X_test,
        labels=y_test,
        time_transforms=None,
        frequency_transforms=None,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return train_loader, validation_loader, test_loader

def main():

    tfc_classifier = build_model()
    train_loader, validation_loader, test_loader = get_data()

    tfc_classifier.train(train_loader)
    val = tfc_classifier.predict(test_loader)

    res = {}
    res['Train dataset'] = dataset
    res['Classifier'] = clsf
    res['Accuracy'] = val

    import yaml
    with results_tfc_classifier.open("w") as f:
        yaml.dump(res, f)

if __name__ == "__main__":
    main()