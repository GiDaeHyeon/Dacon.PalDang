import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler


class SimpleDataset(Dataset):
    def __init__(self, mode: str, time_step: int = 72) -> None:
        super(SimpleDataset, self).__init__()
        self.mode = mode
        data = pd.read_csv('./competition_data/data.csv', encoding='utf8')

        del data['ymdhm']
        y_cols = ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']

        y = data[y_cols]

        for c in y_cols:
            del data[c]
        x = data[['inf', 'tototf', 'fw_1018662', 'fw_1018683', 'fw_1019630']].copy()

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        self.x, self.y = self.create_dataset(x=x, y=y, time_step=time_step)

    def create_dataset(self, x: pd.DataFrame, y: pd.DataFrame, time_step: int = 72) -> tuple:
        if self.mode == 'train':
            min_idx, max_idx = time_step, 210000
        elif self.mode == 'val':
            min_idx, max_idx = 210000 - time_step, 258801
        else:
            min_idx, max_idx = 0, 0

        x_data = []
        for idx in range(min_idx, max_idx + 1):
            if idx - time_step < 0:
                continue
            else:
                x_data.append(x[idx - time_step:idx])

        y_data = []
        for idx in range(min_idx, max_idx + 1):
            y_data.append(y.iloc[idx].values)

        return x_data, y_data

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple:
        try:
            return torch.Tensor(self.x[idx]).float(), torch.Tensor(self.y[idx]).float()
        except IndexError:
            print(idx)
            raise ValueError


class SimpleDataModule(LightningDataModule):
    def __init__(self, time_step: int = 72) -> None:
        super(SimpleDataModule, self).__init__()
        self.trainset = SimpleDataset(mode='train', time_step=time_step)
        self.valset = SimpleDataset(mode='val', time_step=time_step)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.trainset,
                          batch_size=128,
                          num_workers=0,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.valset,
                          batch_size=256,
                          num_workers=0,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=False)
