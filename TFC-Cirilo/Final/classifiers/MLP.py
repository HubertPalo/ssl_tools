import torch
import pytorch_lightning as pl
from typing import Any
from config import Config

from torchmetrics.functional import accuracy

cfg = Config()

class Mlp(pl.LightningModule):
    def __init__(self, num_classes, model):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(2 * 128, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
        self.learning_rate = 1e-3
        self.loss_func = torch.nn.CrossEntropyLoss()
        

    def forward(self, x):
        *_, x = self.model(x)
        emb_flatten = x.reshape(x.shape[0], -1)
        x = self.fc(emb_flatten)
        x = torch.sigmoid(x)
        y = self.fc2(x)
        return y
    
    def configure_optimizers(self) -> Any:
        learnable_parameters = self.parameters()
        optimizer = torch.optim.Adam(learnable_parameters, lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, labels, aug1, data_f, aug1_f = batch
        predictions = self(batch)
        
        loss = self.loss_func(predictions, labels)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx, "validation")

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx, "test")
        
        return {"test_loss": loss, "test_acc": acc}

    def _shared_eval_step(self, batch, batch_idx, stage):
        data, labels, aug1, data_f, aug1_f = batch
        predictions = self(batch)
        
        loss = self.loss_func(predictions, labels)
       
        acc = accuracy(
            torch.argmax(predictions, dim=1),
            labels,
            task="multiclass",
            num_classes=self.num_classes,
        )

        self.log(
            f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss, acc
    
