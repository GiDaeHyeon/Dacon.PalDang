from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from pytorch_lightning import LightningModule


class SimpleTrainModule(LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-4, loss_fn: nn.Module = nn.MSELoss()) -> None:
        super(SimpleTrainModule, self).__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(params=self.model.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: tuple, is_train: bool = True) -> torch.Tensor or tuple:
        x, y = batch
        y_hat = self(x)
        if is_train:
            return self.loss_fn(y, y_hat)
        else:
            r2 = r2_score(y, y_hat)
            rmse = mean_squared_error(y, y_hat) ** .5
            score = rmse / r2
            self.log('r2_score', r2, on_step=False, on_epoch=True)
            self.log('rmse', rmse, on_step=False, on_epoch=True)
            self.log('score', score, on_step=False, on_epoch=True)
            return self.loss_fn(y, y_hat)

    def training_step(self, batch, *args, **kwargs) -> dict:
        loss = self._step(batch=batch)
        self.log('loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, *args, **kwargs) -> Optional[dict]:
        loss = self._step(batch=batch, is_train=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss}
