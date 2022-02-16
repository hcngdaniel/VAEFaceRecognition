#!/usr/bin/env python3
import os
import itertools
import matplotlib.pyplot as plt
import torch.utils.data

import utils
from dataset import CelebALandmarks
from models.model_1 import Encoder, Decoder, LossFunction


# loads the configs
config = utils.load_train_config('train_config.json')

# loads the dataset
print('loading dataset...')
dataset = CelebALandmarks(config.dataset_path)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    shuffle=True,
)

# initialize the model
print('initializing model...')
encoder = Encoder()
decoder = Decoder()
encoder.to(config.device)
decoder.to(config.device)
if os.path.exists(config.encoder_path):
    encoder.load_state_dict(torch.load(config.encoder_path))
if os.path.exists(config.decoder_path):
    decoder.load_state_dict(torch.load(config.decoder_path))
encoder.train()
decoder.train()

# initialize training loss functions and optimizer
loss_function = LossFunction()
optimizer = torch.optim.Adagrad(
    params=itertools.chain(encoder.parameters(), decoder.parameters()),
    lr=config.learning_rate,
)

print('start to train')
plt.ion()
losses = []
reconstruction_losses = []
kl_losses = []
for epoch in range(config.start_epoch, config.end_epoch):
    for batch, x in enumerate(dataloader, 1):
        # normalization
        if batch == len(dataloader):
            continue

        # x preprocess
        x = x.to(config.device)

        # run the model
        # encode
        mean, standard_deviation = encoder(x)
        # add noise
        noise = torch.randn(mean.size()).to(config.device)
        encoded_with_noise = mean + standard_deviation * noise
        # decode
        reconstructed = decoder(encoded_with_noise)

        # calculate loss
        loss, (reconstruction_loss, kl_loss) = loss_function(x, reconstructed, mean, standard_deviation)
        losses.append(loss.item() / x.size(0))
        reconstruction_losses.append(reconstruction_loss.item() / x.size(0))
        kl_losses.append(kl_loss.item() / x.size(0))
        # plot it out
        plt.subplot(221).cla()
        plt.subplot(222).cla()
        plt.subplot(223).cla()
        plt.subplot(221).plot(losses)
        plt.subplot(222).plot(reconstruction_losses)
        plt.subplot(223).plot(kl_losses)
        plt.pause(1e-10)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show the process of training
        print(f'epoch: {epoch} ({batch}/{len(dataloader)}) loss: {loss.item() / x.size(dim=0):.4f}')

    # save
    torch.save(encoder.state_dict(), config.encoder_path)
    torch.save(decoder.state_dict(), config.decoder_path)
plt.ioff()
