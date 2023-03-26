import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.BingRGB import BingRGB
from models.TSS import TSS
from pytorch_lightning.callbacks import ModelCheckpoint
from torchsummary import summary
import torch
from utils import DatasetDownloader, Config

config = Config("./config/config.yaml")

parser = argparse.ArgumentParser()
parser.add_argument('-c','--ckpt_path', type=str, help='Path to checkpoint file', default=None)
parser.add_argument('--init_weight_lr', type=float, help='The value for init_weight_lr.', default=1e-5)
parser.add_argument('--init_weight_momentum', type=float, help='The value for init_weight_momentum.', default=0.9)
parser.add_argument('--lr', type=float, help='The value for lr.', default=.0025)
args = parser.parse_args()

data_module = BingRGB(**config.datasets_config)

model = TSS(lr=args.lr, init_weight_lr=args.init_weight_lr, init_weight_momentum=args.init_weight_momentum, download_config=config.config_dict.datasets.download.weight)

trainer = pl.Trainer(**config.test_config,
logger=[
    CSVLogger(**config.config_dict.trainer.logger.CSVLogger),
])

trainer.test(model, data_module, ckpt_path=args.ckpt_path)