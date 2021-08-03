import visual_models
import datasets

import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DeepSpeedPlugin, DeepSpeedPrecisionPlugin


import numpy as np
from time import time
import os
import argparse
import itertools 

# for debugging
torch.autograd.set_detect_anomaly(True)
vae_model_by_str = {
    'Conv':visual_models.ConvVAE,
    'ResNet':visual_models.ResnetVAE
}

class RewardMLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, learning_rate, scheduler_kwargs, batch_size):#, VAE_path, VAE_class='Conv'):
        super().__init__()
        self.save_hyperparameters()

        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.MSELoss()

        # create linear model
        
        self.model = [nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.GELU())]
        for i in range(len(hidden_dims)-1):
            self.model.append(nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.BatchNorm1d(hidden_dims[i+1]), nn.GELU()))
        self.model.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.ModuleList(self.model)
        
    
    def forward(self, obs):
        out = self.model[0](obs)
        for i in range(1,len(self.model)-1): # skip connections
            out = out + self.model[i](out)
        return self.model[-1](out)
        #return self.model(obs)
    
    def _step(self, batch):
        # unpack batch
        pov, vec, rew = batch
        
        # predict reward
        pred_rew = self.forward(vec)

        # compute loss
        loss = self.loss_fn(pred_rew.squeeze(), rew)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('Training/loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('Validation/loss', loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        # set up optimizer
        optimizer =  AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # set up 
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.hparams.scheduler_kwargs['lr_step_mode'],
            'frequency': self.hparams.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}

def train(env_name, data_dir, lr, val_perc, batch_size, num_data, 
                epochs, lr_gamma, lr_decrease_freq, log_dir, model_class, lr_step_mode,
                val_check_interval, 
                #VAE_class, VAE_path, 
                upsample, backfill, backfill_discount,
                precision, load_from_checkpoint, version_dir, accumulate_grad_batches):
    
    # make sure that relevant dirs exist
    run_name = f'RewardModel_{model_class}/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'Saving logs and model to {log_dir}')

    # set hidden dims and input dim
    # TODO parse these as args
    hidden_dims = [1000,1000,1000,1000,1000]
    input_dim = 192
    input_dim=64

    # instantiate model
    if load_from_checkpoint:
        checkpoint = os.path.join(version_dir, 'checkpoints', 'last.ckpt')        
        print(f'Loading model from {checkpoint}')
        model = RewardMLP.load_from_checkpoint(checkpoint, lr=lr)
    else:
        scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
        model = RewardMLP(input_dim, hidden_dims, lr, scheduler_kwargs, batch_size)#, VAE_path, VAE_class)
        
    

    # load data
    data = datasets.RewardData(env_name, data_dir, num_data, upsample)
    lengths = [len(data)-int(len(data)*val_perc), int(len(data)*val_perc)]
    train_data, val_data = random_split(data, lengths)
    train_sampler = WeightedRandomSampler(train_data.dataset.weights[train_data.indices], 2 * len(train_data.dataset.rewards[train_data.indices][train_data.dataset.rewards[train_data.indices] > 0]), replacement=False)
    #val_sampler = WeightedRandomSampler(val_data.dataset.weights[val_data.indices], 2 * len(val_data.dataset.rewards[val_data.indices][val_data.dataset.rewards[val_data.indices] > 0]), replacement=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=6, pin_memory=True)#, sampler=val_sampler)

    num_batches = len(train_data) // batch_size
    if len(train_data) % batch_size != 0:
        num_batches += 1

    print(f'\nnum train samples = {len(train_data)} --> {num_batches} train batches')
    print(f'num val samples = {len(val_data)}')

    # create callbacks to sample reconstructed images and for model checkpointing
    checkpoint_callback = ModelCheckpoint(mode="min", monitor="Validation/loss", save_last=True)
    trainer=pl.Trainer(
                    precision=precision, #32 is normal, 16 is mixed precision
                    progress_bar_refresh_rate=1, #every N batches update progress bar
                    callbacks=[checkpoint_callback],
                    gpus=torch.cuda.device_count(),
                    default_root_dir=log_dir,
                    max_epochs=epochs,
                    log_every_n_steps=10,
                    val_check_interval=val_check_interval if val_check_interval > 1 else float(val_check_interval)
                    )
                    
    # fit model
    trainer.fit(model, train_loader, val_loader)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--env_name')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_data', default=0, type=int, help='Number of datapoints to use')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--val_perc', default=0.1, type=float, help='How much of the data should be used for validation')
    parser.add_argument('--val_check_interval', default=1, type=int, help='How often to validate. N == 1 --> once per epoch; N > 1 --> every N steps')
    parser.add_argument('--precision', default=32, type=int, help='Numerical precision', choices=[16,32])
    parser.add_argument('--load_from_checkpoint', default=False, action='store_true')
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, help='How many batches to accumulate batches over')
    parser.add_argument('--version_dir', default='', type=str, help='Version directory of model, if training is resumed from checkpoint')
    parser.add_argument('--model_class', default='MLP', type=str)
    parser.add_argument('--upsample', action='store_true')
    parser.add_argument('--backfill', action='store_true')
    parser.add_argument('--backfill_discount', type=float, default=0.95)
    
    # VAE args
    #parser.add_argument('--VAE_class', type=str, default='Conv', choices=['Conv', 'ResNet'])
    #parser.add_argument('--VAE_path', help='Path to encoding model')

    args = vars(parser.parse_args())

    train(**args)
