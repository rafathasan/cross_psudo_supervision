import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from my_datamodule import MyDataModule
from my_model import MyModel

parser = argparse.ArgumentParser()

# add arguments for hyperparameters and other configurations
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training and validation')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate for optimizer')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--save_dir', type=str, default='./checkpoints',
                    help='Directory for saving checkpoints')

args = parser.parse_args()

# create data module
data_module = MyDataModule(batch_size=args.batch_size)

# create model
model = MyModel()

# create logger
logger = TensorBoardLogger('logs/', name='my_experiment')

# create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=args.save_dir,
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

# create trainer
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger=logger, 
                     checkpoint_callback=checkpoint_callback)

# fit model
trainer.fit(model, data_module)

# load best checkpoint
best_checkpoint_path = checkpoint_callback.best_model_path
best_model = MyModel.load_from_checkpoint(best_checkpoint_path)

# evaluate model on test set
test_results = trainer.test(best_model, data_module.test_dataloader())
print(test_results)