#!/usr/bin/env python
import sys
sys.path.append(r'C:\Users\dpalo\Documents\GitHub\ssl_tools')


import lightning as L
import torch

from ssl_tools.experiments import LightningSSLTrain, LightningTest, auto_main
from torchmetrics import Accuracy
from ssl_tools.models.ssl.classifier import SSLDiscriminator
from ssl_tools.models.ssl.modules.heads import TFCPredictionHead, ConvAEPredictionHead
from ssl_tools.models.ssl.tfc import build_tfc_transformer
from ssl_tools.models.ssl.barlowtwins import BarlowTwins
from ssl_tools.data.data_modules import TFCDataModule, UserActivityFolderDataModule, MultiModalHARSeriesDataModule, MultiModalHARSeriesDataModuleForBarlowTwins


class BarlowTwinsTrain(LightningSSLTrain):
    _MODEL_NAME = "BarlowTwins"
    
    def __init__(
            self,
            data: str,
            encoding_size: int = 150,
            in_channel: int = 6,
            input_size = (6,60),
            # Model parameters
            lambda_=5e-3,
            proj_head_size=36,
            proj_head_num=2,
            conv_kernel=3,
            conv_padding=1,
            conv_stride=1,
            conv_num=5,
            dropout=0.3,
            update_backbone: bool = True,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.encoding_size = encoding_size
        self.in_channel = in_channel
        self.input_size = input_size
        self.lambda_ = lambda_
        self.proj_head_size = proj_head_size
        self.proj_head_num = proj_head_num
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding
        self.conv_stride = conv_stride
        self.conv_num = conv_num
        self.dropout = dropout
        self.update_backbone = update_backbone

    def get_pretrain_model(self) -> L.LightningModule:
        model = BarlowTwins(
            encoding_size=self.encoding_size,
            lambda_=self.lambda_,
            proj_head_size=self.proj_head_size,
            proj_head_num=self.proj_head_num,
            conv_kernel=self.conv_kernel,
            conv_padding=self.conv_padding,
            conv_stride=self.conv_stride,
            conv_num=self.conv_num,
            dropout=self.dropout,
        )
        print("Model:::::::::", model)
        return model
    
    def get_pretrain_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModuleForBarlowTwins(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
        )
        return data_module
    
    def get_finetune_model(
        self, load_backbone: str = None
    ) -> L.LightningModule:
        model = self.get_pretrain_model()

        # print("Loading backbone model...", model)
        
        if load_backbone is not None:
            self.load_checkpoint(model, load_backbone)

        classifier = ConvAEPredictionHead(
                input_dim=150,
                hidden_dim1=150,
                hidden_dim2=128,
                output_dim=self.num_classes,
            )

        task = "multiclass" if self.num_classes > 2 else "binary"
        # print("Classifier:::::::::", classifier)
        model = SSLDiscriminator(
            backbone=model.encoder,
            head=classifier,
            loss_fn=torch.nn.CrossEntropyLoss(),
            learning_rate=self.learning_rate,
            metrics={"acc": Accuracy(task=task, num_classes=self.num_classes)},
            update_backbone=self.update_backbone,
        )
        return model
    
    def get_finetune_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
        )
        return data_module
    
if __name__ == "__main__":
    options = {
        "fit": BarlowTwinsTrain,
        # "test": CPCTest,
    }
    auto_main(options)