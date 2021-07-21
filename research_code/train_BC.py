import env_wrappers
import models
import datasets

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import argparse
import os
from pyvirtualdisplay import Display
import gym


def train_bc(model_path, data_dir, env_name, lr, val_perc, eval_freq, batch_size, epochs, lr_gamma, lr_decrease_freq, log_dir, lr_step_mode, n_clusters, centroids_path):

    # make sure that relevant dirs exist
    run_name = f'BCLinear/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'Saving logs and model to {log_dir}')

    # load data
    data = datasets.BehavCloneData(env_name, data_dir)
    lengths = [len(data)-int(len(data)*val_perc), int(len(data)*val_perc)]
    train_data, val_data = random_split(data, lengths)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=3)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=3)


    num_batches = len(train_data) // batch_size
    if len(train_data) % batch_size != 0:
        num_batches += 1

    print(f'\nnum train samples = {len(train_data)} --> {num_batches} train batches')
    print(f'num val samples = {len(val_data)}')


    # init model
    hidden_dims = [512, 512, 512, 512, 512]
    input_dim = 192
    output_dim = n_clusters
    scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}

    model = models.BCLinear(input_dim, hidden_dims, output_dim, lr, scheduler_kwargs, centroids_path, model_path)


    # create callbacks to sample reconstructed images and for model checkpointing
    checkpoint_callback = ModelCheckpoint(mode="min", monitor="Validation/loss", save_last=True)

    # create trainer
    trainer=pl.Trainer(
                    precision=16, #32 is normal, 16 is mixed precision
                    progress_bar_refresh_rate=100, #every N batches update progress bar
                    callbacks=[checkpoint_callback],
                    gpus=torch.cuda.device_count(),
                    auto_lr_find=True,
                    #auto_scale_batch_size='binsearch',
                    accelerator='dp', #anything else here seems to lead to crashes/errors
                    default_root_dir=log_dir,
                    max_epochs=epochs
                )
    trainer.logger._default_hp_metric = None # optional logging metric that we don't need right now
    trainer.fit(model, train_loader, val_loader) # fit model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to encoding model')
    parser.add_argument('--data_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--env_name')
    parser.add_argument('--centroids_path', help='Path to file containing action centroids')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--val_perc', default=0.1, type=float, help='How much of the data should be used for validation')
    parser.add_argument('--eval_freq', default=1, type=int, help='How often to reconstruct a random val image for tensorboard')
    parser.add_argument('--n_clusters', default=200, type=int)

    args = vars(parser.parse_args())

    train_bc(**args)
