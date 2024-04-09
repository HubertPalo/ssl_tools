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
from ssl_tools.models.ssl.convae import ConvAE
from ssl_tools.data.data_modules import TFCDataModule, UserActivityFolderDataModule, MultiModalHARSeriesDataModule
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class ConvAETrain(LightningSSLTrain):
    _MODEL_NAME = "ConvAE"
    
    def __init__(
            self,
            data: str,
            encoding_size: int = 150,
            in_channel: int = 6,
            # Model parameters
            conv_num=3, fc_num=2, #enc_size=150,
            conv_kernel=3, conv_padding=0, conv_stride=1,
            conv_groups=1, conv_dilation=1,
            pooling_type='none', pooling_kernel=2, pooling_stride=2,
            dropout=0.2, input_size=(6,60),
            pad_length: bool = False,
            num_classes: int = 6,
            update_backbone: bool = True,
            window_size: int = 60,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.encoding_size = encoding_size
        self.in_channel = in_channel
        self.conv_num = conv_num
        self.fc_num = fc_num
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding
        self.conv_stride = conv_stride
        self.conv_groups = conv_groups
        self.conv_dilation = conv_dilation
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.pooling_stride = pooling_stride
        self.dropout = dropout
        self.input_size = input_size
        self.num_classes = num_classes
        self.update_backbone = update_backbone
        self.pad_length = pad_length
        self.window_size = window_size
        
    def get_pretrain_model(self) -> L.LightningModule:
        model = ConvAE(
            conv_num=self.conv_num, fc_num=self.fc_num, enc_size=self.encoding_size,
            conv_kernel=self.conv_kernel, conv_padding=self.conv_padding, conv_stride=self.conv_stride,
            conv_groups=self.conv_groups, conv_dilation=self.conv_dilation,
            pooling_type=self.pooling_type, pooling_kernel=self.pooling_kernel, pooling_stride=self.pooling_stride,
            dropout=self.dropout, input_size=self.input_size,
            # lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        return model
    
    def get_pretrain_data_module(self) -> L.LightningDataModule:
        # data_module = UserActivityFolderDataModule(
        #     data_path=self.data,
        #     batch_size=self.batch_size,
        #     pad=self.pad_length,
        #     num_workers=self.num_workers,
        # )
        # return data_module
        data_module = MultiModalHARSeriesDataModule(
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
        "fit": ConvAETrain,
        # "test": CPCTest,
    }
    auto_main(options)