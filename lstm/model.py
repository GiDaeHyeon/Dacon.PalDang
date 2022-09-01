import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = .5) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(),
            nn.Dropout(.5),
            nn.Linear(64, 32),
            nn.LeakyReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(32, 4),
            nn.ReLU()
        )

    def forward(self, x) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.output_layer(x)[:, -1]
