import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.TSS import TSS
from torchsummary import summary
import torch
import argparse
import os
from utils import DatasetDownloader, Config
from datasets import BingRGB

config = Config("./config/config.yaml")

parser = argparse.ArgumentParser()
parser.add_argument('-c','--ckpt_path', type=str, help='Path to checkpoint file', default=None)
parser.add_argument('--init_weight_lr', type=float, help='The value for init_weight_lr.', default=1e-5)
parser.add_argument('--init_weight_momentum', type=float, help='The value for init_weight_momentum.', default=0.9)
parser.add_argument('--lr', type=float, help='The value for lr.', default=.0025)
args = parser.parse_args()

data_module = BingRGB(**config.datasets_config)

model = TSS(lr=args.lr, init_weight_lr=args.init_weight_lr, init_weight_momentum=args.init_weight_momentum, download_config=config.config_dict.datasets.download.weight)

trainer = pl.Trainer(**config.train_config,
callbacks=[
    ModelCheckpoint(**config.config_dict.trainer.callbacks.ModelCheckpoint)
],
logger=[
    # TensorBoardLogger(**config.config_dict.trainer.logger.TensorBoardLogger),
    # CSVLogger(**config.config_dict.trainer.logger.CSVLogger),
    WandbLogger(**config.config_dict.trainer.logger.WandbLogger)
])

trainer.fit(model, data_module, ckpt_path=args.ckpt_path)