import models
import datasets

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
from time import time
import os
import argparse

# for debugging
torch.autograd.set_detect_anomaly(True)


def train_DynamicsModel(env_name, data_dir, lr, val_perc, eval_freq, batch_size, epochs, lr_gamma, lr_decrease_freq, log_dir, lr_step_mode, model_path):
    
    # make sure that relevant dirs exist
    run_name = f'DynamicsModel/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'Saving logs and model to {log_dir}')

    ## some model kwargs
    seq_len = 4
    hidden_dims = [512,512,512]
    base_model_class = models.DynamicsBaseModel
    base_model_kwargs = {'input_dim':256, 'hidden_dims':hidden_dims}
    ##

    # init model
    optim_kwargs = {'lr':lr}
    scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
    model = models.NODEDynamicsModel(base_model_class, base_model_kwargs, model_path, optim_kwargs, scheduler_kwargs, seq_len)

    # load data
    data = datasets.DynamicsData(env_name, data_dir, seq_len)
    lengths = [len(data)-int(len(data)*val_perc), int(len(data)*val_perc)]
    train_data, val_data = random_split(data, lengths)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=3)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=3)

    num_batches = len(train_data) // batch_size
    if len(train_data) % batch_size != 0:
        num_batches += 1

    print(f'\nnum train samples = {len(train_data)} --> {num_batches} train batches')
    print(f'num val samples = {len(val_data)}')


    trainer=pl.Trainer(
                    precision=32, #32 is normal, 16 is mixed precision
                    progress_bar_refresh_rate=100, #every N batches update progress bar
                    checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="Validation/loss"),
                    gpus=torch.cuda.device_count(),
                    accelerator='dp', #anything else here seems to lead to crashes/errors
                    default_root_dir=log_dir,
                    max_epochs=epochs
                )
    trainer.logger._default_hp_metric = None # optional logging metric that we don't need right now
    trainer.fit(model, train_loader, val_loader)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', help='Path to encoding model')
    parser.add_argument('--data_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--env_name')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--val_perc', default=0.1, type=float, help='How much of the data should be used for validation')
    parser.add_argument('--eval_freq', default=1, type=int, help='How often to reconstruct a random val image for tensorboard')

    args = vars(parser.parse_args())

    train_DynamicsModel(**args)
