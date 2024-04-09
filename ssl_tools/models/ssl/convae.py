import torch
import lightning as L
import numpy as np
import pdb
from numpy import linspace

from ssl_tools.utils.configurable import Configurable
from ssl_tools.models.layers.gru import GRUEncoder
from ssl_tools.models.layers.conv import CNNEncoder as CNN

import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class ConvAE(L.LightningModule, Configurable):

    def __init__(
            self, conv_num=1, fc_num=1, enc_size=32,
            conv_kernel=5, conv_padding=0, conv_stride=1,
            conv_groups=1, conv_dilation=1,
            pooling_type='none', pooling_kernel=2, pooling_stride=2,
            dropout=0.2, input_size=(6,60),
            lr: float = 1e-3, weight_decay: float = 0.0,
        ):
        super().__init__()
        self.conv_num = conv_num
        self.fc_num = fc_num
        self.enc_size = enc_size
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
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.loss_function = nn.MSELoss()
        # print(self.get_config())
        print(self.encoder)
        print(self.decoder)

    def build_encoder(self):
        def conv_l_out(l_in, kernel, padding, stride, dilation=1):
            return int(((l_in + 2*padding - dilation*(kernel-1) - 1)/stride) + 1)
        def pool_l_out(l_in, kernel, stride):
            return int((l_in - kernel)/stride + 1)
        conv_out_channels_dict = {
            1: [128],
            2: [64, 128],
            3: [64, 128, 256],
            4: [32, 64, 128, 256],
            5: [16, 32, 64, 128, 256],
            6: [8, 16, 32, 64, 128, 256]
        }
        pooling_types_dict = {
            'max': nn.MaxPool1d,
            'avg': nn.AvgPool1d
        }
        layers = []
        out_channels_list = conv_out_channels_dict.get(self.conv_num, [])
        in_channels = self.input_size[0]
        l_in = self.input_size[1]
        # Adding the Conv layers
        for out_channels in out_channels_list:
            l_out = conv_l_out(l_in, self.conv_kernel, self.conv_padding, self.conv_stride, self.conv_dilation)
            if l_out < 1: break
            new_conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding,
                dilation=self.conv_dilation,
                groups=self.conv_groups
            )
            layers.append(new_conv_layer)
            in_channels = out_channels
            l_in = l_out
            # Adding the Pooling layers
            if self.pooling_type != 'none':
                l_out = pool_l_out(l_in, self.pooling_kernel, self.pooling_stride)
                if l_out < 1: break
                pool_layer = pooling_types_dict[self.pooling_type](self.pooling_kernel, self.pooling_stride)
                layers.append(pool_layer)
                l_in = l_out
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
        if out_channels_list != []:
            # Adding the View layer
            layers.append(View((-1, in_channels*l_in)))
        # Adding the FC layers 
        dimensions = linspace(in_channels*l_in, self.enc_size, self.fc_num+2).round().astype(int)
        for index, dim in enumerate(dimensions[:-1]):
            layer = nn.Linear(dim, dimensions[index+1])
            layers.append(layer)
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
        # Deleting the last ReLU and Dropout
        layers.pop()
        layers.pop()
        return nn.Sequential(*layers)
    
    def build_decoder(self):
        def conv_l_out(l_in, kernel, padding, stride, dilation=1):
            l_out = int(((l_in + 2*padding - dilation*(kernel-1) - 1)/stride) + 1)
            output_padding = l_out - (l_in-1)*stride + 2*padding + dilation*(kernel-1) - 1
            return l_out, output_padding
        def pool_l_out(l_in, kernel, stride):
            return int((l_in - kernel)/stride + 1)
        conv_out_channels_dict = {
            1: [128],
            2: [64, 128],
            3: [64, 128, 256],
            4: [32, 64, 128, 256],
            5: [16, 32, 64, 128, 256],
            6: [8, 16, 32, 64, 128, 256]
        }
        pooling_types_dict = {
            'max': nn.MaxPool1d,
            'avg': nn.AvgPool1d
        }
        layers = []
        out_channels_list = conv_out_channels_dict.get(self.conv_num, [])
        in_channels = self.input_size[0]
        l_in = self.input_size[1]
        # Simulating Conv layers
        for out_channels in out_channels_list:
            l_out, output_padding = conv_l_out(l_in, self.conv_kernel, self.conv_padding, self.conv_stride, self.conv_dilation)
            if l_out < 1: break
            # print(f"l_out: {l_out}, output_padding: {output_padding}")
            new_conv_layer = nn.ConvTranspose1d(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding,
                dilation=self.conv_dilation,
                groups=self.conv_groups,
                output_padding=output_padding
            )
            layers.append(new_conv_layer)
            in_channels = out_channels
            l_in = l_out
            # Adding the Pooling layers
            if self.pooling_type != 'none':
                l_out = pool_l_out(l_in, self.pooling_kernel, self.pooling_stride)
                if l_out < 1: break
                pool_layer = pooling_types_dict[self.pooling_type](self.pooling_kernel, self.pooling_stride)
                layers.append(pool_layer)
                l_in = l_out
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        if out_channels_list != []:
            # Deleting the last ReLU and Dropout
            layers.pop()
            layers.pop()
            # Adding the View layer
            layers.append(View((-1, in_channels, l_in)))
        # Adding the FC layers 
        dimensions = linspace(in_channels*l_in, self.enc_size, self.fc_num+2).round().astype(int)
        for index, dim in enumerate(dimensions[:-1]):
            layer = nn.Linear(dimensions[index+1], dim)
            layers.append(layer)
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
        # Deleting the last ReLU and Dropout
        layers.pop()
        layers.pop()
        # Reverse the layers array
        layers = layers[::-1]
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(sample))
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed = self(x)
        loss = self.loss_function(x_reconstructed, x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed = self(x)
        loss = self.loss_function(x_reconstructed, x)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def get_config(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "conv_num": self.conv_num,
            "fc_num": self.fc_num,
            "enc_size": self.enc_size,
            "conv_kernel": self.conv_kernel,
            "conv_padding": self.conv_padding,
            "conv_stride": self.conv_stride,
            "conv_groups": self.conv_groups,
            "conv_dilation": self.conv_dilation,
            "pooling_type": self.pooling_type,
            "pooling_kernel": self.pooling_kernel,
            "pooling_stride": self.pooling_stride,
            "dropout": self.dropout,
            "input_size": self.input_size
        }