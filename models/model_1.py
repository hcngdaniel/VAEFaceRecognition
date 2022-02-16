#!/usr/bin/env python3
import typing
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(478 * 2, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
        )
        self.mean = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        self.standard_deviation = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) \
            -> typing.Union[typing.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features = self.features(x)
        mean = self.mean(features)
        standard_deviation = torch.exp(self.standard_deviation(features) * 0.5)
        return mean, standard_deviation


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, 478 * 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LossFunction:
    mse = nn.MSELoss()

    def __init__(self):
        pass

    @classmethod
    def reconstruction_loss(cls, h, y):
        return cls.mse(h, y)

    @classmethod
    def kl_loss(cls, mean, standard_deviation):
        return torch.mean((torch.square(mean) + torch.exp(standard_deviation) - 1 - standard_deviation)) / 2

    def __call__(self, y, h, mean, standard_deviation):
        reconstruction_loss = self.reconstruction_loss(h, y)
        kl_loss = self.kl_loss(mean, standard_deviation)
        return reconstruction_loss + kl_loss, (reconstruction_loss, kl_loss)
