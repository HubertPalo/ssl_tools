from torch import optim, nn
import lightning as L
from torch.nn import Conv1d, AdaptiveAvgPool1d, Dropout, ReLU, Linear, BatchNorm1d, Flatten
import torch
from ssl_tools.utils.configurable import Configurable


class BarlowTwins(L.LightningModule, Configurable):
    def __init__(
            self,
            encoding_size=36,
            lambda_=5e-3,
            proj_head_size=36,
            proj_head_num=2,
            conv_kernel=3,
            conv_padding=1,
            conv_stride=1,
            conv_num=5,
            dropout=0.25
            ):
        super().__init__()
        self.encoding_size = encoding_size
        self.proj_head_size = proj_head_size
        self.proj_head_num = proj_head_num
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding
        self.conv_stride = conv_stride
        self.conv_num = conv_num
        self.dropout = dropout
        self.lambda_ = lambda_

        backbone_conv_out_channels = {
            0: [],
            1: [512],
            2: [256, 512],
            3: [128, 256, 512],
            4: [64, 128, 256, 512],
            5: [32, 64, 128, 256, 512],
            6: [16, 32, 64, 128, 256, 512]
        }
        backbone = []
        for i in range(conv_num):
            in_channels = 6 if i == 0 else backbone_conv_out_channels[conv_num][i-1]
            out_channels = backbone_conv_out_channels[conv_num][i]
            backbone.append(Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding))
            backbone.append(ReLU())
            backbone.append(Dropout(dropout))
        backbone.append(AdaptiveAvgPool1d(1))
        backbone.append(Flatten())
        backbone.append(Linear(backbone_conv_out_channels[conv_num][-1], encoding_size))
        projection_head = []
        for i in range(proj_head_num):
            in_channels = encoding_size if i == 0 else proj_head_size
            out_channels = proj_head_size
            projection_head.append(Linear(in_channels, out_channels))
            projection_head.append(BatchNorm1d(out_channels))
            projection_head.append(ReLU())
        projection_head.append(Linear(proj_head_size, proj_head_size))
        
        self.backbone = nn.Sequential(*backbone)
        self.projection_head = nn.Sequential(*projection_head)
        # backbone = nn.Sequential(
        #     Conv1d(in_channels=6, out_channels=32, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
        #     ReLU(),
        #     Dropout(dropout),
        #     Conv1d(in_channels=32, out_channels=64, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
        #     ReLU(),
        #     Dropout(dropout),
        #     Conv1d(in_channels=64, out_channels=128, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
        #     ReLU(),
        #     Dropout(dropout),
        #     Conv1d(in_channels=128, out_channels=256, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
        #     ReLU(),
        #     Dropout(dropout),
        #     Conv1d(in_channels=256, out_channels=512, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
        #     ReLU(),
        #     Dropout(dropout),
        #     AdaptiveAvgPool1d(1),
        #     Flatten(),
        #     Linear(512, encoding_size)
        # )
        # projection_head = nn.Sequential(
        #     Linear(encoding_size, projection_head_dim),
        #     BatchNorm1d(projection_head_dim),
        #     ReLU(),
        #     Linear(projection_head_dim, projection_head_dim),
        #     BatchNorm1d(projection_head_dim),
        #     ReLU(),
        #     Linear(projection_head_dim, projection_head_dim)
        # )
        # self.lambda_ = lambda_
        # self.backbone = backbone
        # self.projection_head = projection_head

    def forward(self, x):
        # use forward for inference/predictions
        x = self.backbone(x)
        x = self.projection_head(x)
        return x
    
    def compute_loss(self, x1, x2):
        # print(len(x1), len(x2))
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)
        # NORMALIZATION
        combined = torch.stack([p1, p2], dim=0)
        normalized = nn.functional.batch_norm(
            combined,
            running_mean=None,
            running_var=None,
            training=True,
            weight=None,
            bias=None
        ).view_as(combined)
        p1_norm, p2_norm = normalized[0], normalized[1]
        N = p1.size(0)
        # CROSS-CORRELATION
        c = p1_norm.T @ p2_norm
        c.div_(N)
        invariance_loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        redundancy_reduction_loss = c.flatten()[:-1].view(c.shape[0]-1, c.shape[0]+1)[:,1:].flatten().pow(2).sum()
        return invariance_loss, redundancy_reduction_loss

    def training_step(self, batch, batch_idx):
        # print("Batch:::::::::", batch[0].shape)
        all_xs = batch[0]
        x1 = all_xs[0]
        x2 = all_xs[1]
        invariance_loss, redundancy_reduction_loss = self.compute_loss(x1, x2)
        loss = invariance_loss + self.lambda_ * redundancy_reduction_loss
        self.log("train_loss", loss)
        self.log("invariance_loss", invariance_loss)
        self.log("redundancy_reduction_loss", redundancy_reduction_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # print("Batch:::::::::", len(batch[0]))
        all_xs = batch[0]
        x1 = all_xs[0]
        x2 = all_xs[1]
        invariance_loss, redundancy_reduction_loss = self.compute_loss(x1, x2)
        loss = invariance_loss + self.lambda_ * redundancy_reduction_loss
        self.log("val_loss", loss)
        self.log("invariance_loss", invariance_loss)
        self.log("redundancy_reduction_loss", redundancy_reduction_loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer
    
    def get_config(self) -> dict:
        return {
            "encoding_size": self.encoding_size,
            "lambda_": self.lambda_,
            "proj_head_size": self.proj_head_size,
            "proj_head_num": self.proj_head_num,
            "conv_kernel": self.conv_kernel,
            "conv_padding": self.conv_padding,
            "conv_stride": self.conv_stride,
            "conv_num": self.conv_num,
            "dropout": self.dropout
        }