import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data.BDDataModule import BDDataModule
from models.TSS import TSS
from pytorch_lightning.callbacks import ModelCheckpoint

model = TSS()
data_module = BDDataModule(3)

save_dir = "/src/outputs/"

# create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath=save_dir,
    filename='model-{epoch:02d}-{train_loss:.2f}',
    save_top_k=3,
    mode='min'
)

# checkpoint_callback.on_validation_end(trainer=None, pl_module=model)
best_model_path = checkpoint_callback.best_model_path
# best_model = TSS.load_from_checkpoint(best_model_path)

print(best_model_path)