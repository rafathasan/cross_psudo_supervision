#!/opt/conda/bin/python

import click
from torchsummary import summary
from utils import download_data, Config
import setproctitle
import os
setproctitle.setproctitle(os.uname()[1])
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.TSS import TSS
import torch
from datasets import BingRGB

@click.group()
def cli():
    pass

@cli.command()
@click.option('--project-name', type=str, default="cps")
@click.option('--lr', default=0.0025, help='Learning rate')
@click.option('--wandb-key', type=str, default=os.getenv('WANDB_KEY'))
@click.option('--model-path', type=str, default='./checkpoints/last.ckpt')
@click.option('--save-dir', type=str, default='./checkpoints')
@click.option('--log-dir', type=str, default='./logs')
@click.option('--config-path', type=click.Path(exists=True), default='./config/config.yaml')

def train(project_name, lr, wandb_key, model_path, save_dir, log_dir, config_path):
    """Train a model on the specified dataset and save it to the specified path."""

    config = Config(config_path)

    wandb.login(key=wandb_key)

    if not os.path.exists(model_path):
        model_path = None
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    data_module = BingRGB(**config.datasets_config)
    data_module.prepare_data()

    model = TSS(lr=lr, cps_scale=1.5)

    trainer = pl.Trainer(
    **config.train_config,
    callbacks=[
        ModelCheckpoint(**config.config_dict.trainer.callbacks.ModelCheckpoint)
    ],
    logger=[
        # TensorBoardLogger(**config.config_dict.trainer.logger.TensorBoardLogger),
        CSVLogger(name=project_name, save_dir=log_dir),
        WandbLogger(project="8th_cps", name=project_name, save_dir=log_dir),
    ])

    trainer.fit(model, data_module, ckpt_path=model_path)

@cli.command()
@click.option('--data-path', type=click.Path(exists=True), default='./data/BingRGB')
@click.option('--model-path', type=click.Path(exists=True), default='./checkpoints/last.ckpt')
@click.option('--log-dir', type=str, default='./logs')
def test(data_path, model_path, log_dir):
    """Test a model on the specified dataset."""

    if os.path.isfile(model_path):
        # If `model_path` is a file, directly pass it to `trainer.test()`
        model_paths = [model_path]
    elif os.path.isdir(model_path):
        # If `model_path` is a directory, find all .ckpt files in the directory
        model_paths = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.ckpt')]
    else:
        raise ValueError(f"Invalid `model_path`: {model_path}")


    config = Config("./config/config.yaml")

    data_module = BingRGB(**config.datasets_config)

    for model_path in model_paths:
        ckpt_name = os.path.splitext(os.path.basename(model_path))[0]

        trainer = pl.Trainer(**config.test_config,
                             logger=[
                                 CSVLogger(save_dir=log_dir, name="cps", version=ckpt_name),
                             ],
                             enable_model_summary=False,
                             )

        # Load the model checkpoint and pass it to `trainer.test()`
        model = TSS.load_from_checkpoint(model_path)
        trainer.test(model=model, datamodule=data_module, ckpt_path=model_path)

if __name__ == '__main__':
    cli()