import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data.BDDataModule import BDDataModule
from models.TSS import TSS
from pytorch_lightning.callbacks import ModelCheckpoint

model = TSS()
data_module = BDDataModule(3)


logger = TensorBoardLogger('logs/', name='my_experiment')
logger2 = CSVLogger("logs", name="my_exp_name")
# ddp_plugin = DDPPlugin(find_unused_parameters=False)

save_dir = "/src/outputs/"

# create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath=save_dir,
    filename='model-{epoch:02d}-{train_loss:.2f}',
    save_top_k=3,
    mode='min',
    save_last=1,
)

trainer = pl.Trainer(
    devices=-1,
    # num_nodes=1,
    # accelerator='cuda',
    # strategy='ddp_cpu',
    max_epochs=30,
    # fast_dev_run=1,
    # logger=[logger, logger2],
    # log_every_n_steps=10,
    # callbacks=[checkpoint_callback],
    )

# trainer.fit(model, data_module)

print(checkpoint_callback.best_model_path)