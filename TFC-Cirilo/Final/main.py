import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

from pathlib import Path

from config import Config
from architectures.Transformer import *
from architectures.CNN1D import *
from dataPreProcess import DataPre
from TFCmodel import TFC
from classifiers.MLP import Mlp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help = "Dataset utilizado para treino")
parser.add_argument('--mode', type=str, help = "modo: pt -> Pretreino, cls -> Classificar, amb -> Ambos")

args = parser.parse_args()

dataset = args.dataset
mode = args.mode

cfg = Config()

def preTrain_mode():
    architecture = CNN_Enconder()
    tfc_model = TFC(architecture)
    dataPre = DataPre(dataset, architecture)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.tfc_pretrain_checkpoint,
        filename= "PreTrainCNN:" + dataset,
        monitor="val_loss",
        save_last=False,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.num_gpus,
        callbacks=[checkpoint_callback],
        strategy=cfg.strategy
    )

    validation_loader = dataPre.get_data("validation")
    train_loader = dataPre.get_data("train")
    
    trainer.fit(tfc_model, train_loader, validation_loader)

def classifier_mode():
    model = TFC(CNN_Enconder())
    # model.load_from_checkpoint("save_models/PreTrainFinal:" + str(dataset) + ".ckpt")
    # model = torch.load("save_models/PreTrainFinal:" + str(dataset) + ".ckpt")
    model.load_state_dict(torch.load("save_models/PreTrainCNN:" + str(dataset) + ".ckpt")["state_dict"])
    print(model.architecture)
    classifier = Mlp(7, model)
    dataPre = DataPre(dataset, model.architecture)

    # Freezing
    # for param in model.parameters():
    #     param.requires_grad = False
    
    train_loader = dataPre.get_data("train") 
    validation_loader = dataPre.get_data("validation") 
    test_loader = dataPre.get_data("test")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_last=False,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.num_gpus,
        callbacks=[checkpoint_callback],
        strategy=cfg.strategy
    )

    print("Fit classifier")
    trainer.fit(classifier, train_loader, validation_loader)

    print("Test classifier")
    res = trainer.test(classifier, test_loader)
    
    print(res)

    res[0]['Train dataset'] = dataset

    results_path = Path("results/" + dataset + "_CNN.yaml")
    print(results_path) 

    import yaml
    with results_path.open("w") as f:
        yaml.dump(res, f)
    
    pass

def both_mode():
    preTrain_mode()
    classifier_mode()

def main():
    if mode == "pt":
        preTrain_mode()
    elif mode == "cls":
        classifier_mode()
    else:
        both_mode()
    

if __name__ == "__main__":
    main()