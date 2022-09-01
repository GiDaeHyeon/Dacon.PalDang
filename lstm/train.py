from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from train_module import SimpleTrainModule
from data_module import SimpleDataModule
from model import LSTMModel


TIME_STEP = 144
INPUT_SIZE = 5

model = LSTMModel(INPUT_SIZE, TIME_STEP)
train_module = SimpleTrainModule(model=model)
data_module = SimpleDataModule(time_step=TIME_STEP)

logger = TensorBoardLogger(save_dir='./logs/SimpleNetwork')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=True, mode='min')

trainer = Trainer(max_epochs=500, callbacks=[early_stopping], logger=logger)


if __name__ == '__main__':
    trainer.fit(train_module, data_module)
