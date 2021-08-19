import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import os
import argparse
import einops
from copy import deepcopy

import vqvae
import datasets

class PretrainQNetwork(pl.LightningModule):
    
    def __init__(self, vqvae_path, n_actions, optim_kwargs, scheduler_kwargs, target_update_rate, margin, gamma, n):
        self.save_hyperparameters()
        
        # load VAE
        self.VAE = vqvae.VQVAE.load_from_checkpoint(self.hparams.vqvae_path)
        
        # conv feature extractor
        dummy, dummy_idcs, _ = self.VAE.encode_only(torch.ones(2,3,64,64).float().to(self.VAE.device))
        num_channels = dummy.shape[1]
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=256, kernel_size=3, padding=1, stride=2), # 16 -> 8
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2), # 8 -> 4
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2), # 4 -> 2
            nn.GELU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, stride=2)#, # 2 -> 1
        )
        
        # q network
        self.q_net = nn.Sequential(
            nn.Linear(2048 + 64, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.hparams.n_actions)
        )
        
        # init target net
        self._update_target()
        
        # loss function
        self.loss_fn = nn.MSELoss()
    
    def _update_target(self):
        self.target_net = nn.ModuleDict({
            'conv_net':deepcopy(self.conv_net),
            'q_net':deepcopy(self.q_net)
        })
        
    def forward(self, pov, vec_obs, target=False):
        # encode into vqvae latent
        out, _, _ = self.VAE(pov)
        if target:
            # apply conv feature ext
            out = einops.rearrange(self.target_net['conv_net'](out), 'b c h w -> b (c h w)')
            # apply q network
            out = self.target_net['q_net'](torch.cat([out, vec_obs], dim=1))
        else:
            # apply conv feature ext
            out = einops.rearrange(self.conv(out), 'b c h w -> b (c h w)')
            # apply q network
            out = self.q_net(torch.cat([out, vec_obs], dim=1))
        return out
    
    def _large_margin_classification_loss(self, q_values, expert_action):
        '''
        Computes the large margin classification loss J_E(Q) from the DQfD paper
        '''
        q_values = q_values + self.hparams.margin
        q_values[:,expert_action] = q_values[:,expert_action] - self.hparams.margin
        return torch.max(q_values, dim=1)[0] - q_values[expert_action]
    
    def training_step(self, batch, batch_idx):
        pov, vec_obs, action, reward, next_pov, next_vec_obs, n_step_reward, n_step_pov, n_step_vec_obs = batch
        
        # predict q values
        pred = self(pov, vec_obs)
        next_pred = self(next_pov, next_vec_obs, target=True).detach()
        n_step_pred = self(n_step_pov, n_step_vec_obs, target=True).detach()
        
        # compute the individual losses
        classification_loss = self._large_margin_classification_loss(pred, action)
        one_step_loss = self.loss_fn(torch.max(pred, dim=1)[0], reward + self.hparams.gamma * torch.max(next_pred, dim=1)[0])
        n_step_loss = self.loss_fn(torch.max(pred, dim=1)[0], n_step_reward + (self.hparams.gamma ** self.hparams.n) * torch.max(n_step_pred, dim=1)[0])
        
        # sum up losses
        loss = classification_loss + one_step_loss + n_step_loss
        
        # logging
        self.log('Training/Loss', loss, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        pov, vec_obs, action, reward, next_pov, next_vec_obs, n_step_reward, n_step_pov, n_step_vec_obs = batch
        
        # predict q values
        pred = self(pov, vec_obs)
        next_pred = self(next_pov, next_vec_obs, target=True).detach()
        n_step_pred = self(n_step_pov, n_step_vec_obs, target=True).detach()
        
        # compute the individual losses
        classification_loss = self._large_margin_classification_loss(pred, action)
        one_step_loss = self.loss_fn(torch.max(pred, dim=1)[0], reward + self.hparams.gamma * torch.max(next_pred, dim=1)[0])
        n_step_loss = self.loss_fn(torch.max(pred, dim=1)[0], n_step_reward + (self.hparams.gamma ** self.hparams.n) * torch.max(n_step_pred, dim=1)[0])
        
        # sum up losses
        loss = classification_loss + one_step_loss + n_step_loss
        
        # logging
        self.log('Validation/Loss', loss, on_epoch=True)
        
        return loss
    
    def on_after_backward(self):
        if (self.global_step + 1) % self.hparams.target_update_rate == 0:
            print(f'Global step {self.global_step+1}: Updating Target Network')
            self._update_target()
        
    def configure_optimizers(self):
        # set up optimizer
        params = list(self.q_net.parameters()) + list(self.conv_net.parameters())
        optimizer = torch.optim.AdamW(params, **self.hparams.optim_kwargs)
        # set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.hparams.scheduler_kwargs['lr_step_mode'],
            'frequency': self.hparams.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}


def main(env_name, batch_size, lr, weight_decay, load_from_checkpoint, version, vqvae_path, data_dir, log_dir,
        num_data, epochs, lr_gamma, lr_step_mode, lr_decrease_freq, val_perc, val_check_interval,
        centroids_path, target_update_rate, margin, gamma, n):
    # make sure that relevant dirs exist
    run_name = f'PretrainQNetwork/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    # load centroids
    print(f'\nLoading centroids from {centroids_path}...')
    centroids = np.load(centroids_path)
    print(f'Loaded centroids! Shape is {centroids.shape}.')
    
    ## some model kwargs
    optim_kwargs = {'lr':lr, 'weight_decay':weight_decay}
    scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
    model_kwargs = {
        'vqvae_path':vqvae_path,
        'optim_kwargs':optim_kwargs,
        'scheduler_kwargs':scheduler_kwargs,
        'n_actions':centroids.shape[0],
        'target_update_rate':target_update_rate,
        'margin':margin,
        'gamma':gamma,
        'n':n
    }

    if load_from_checkpoint:
        checkpoint_file = os.path.join(log_dir, 'lightning_logs', f'version_{version}', 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint_file}')
        model = PretrainQNetwork.load_from_checkpoint(checkpoint_file, **model_kwargs)
    else:
        model = PretrainQNetwork(**model_kwargs)
        
    
    # load data
    data = datasets.PretrainQNetData(env_name, data_dir, centroids, n, gamma, num_data)
    lengths = [len(data)-int(len(data)*val_perc), int(len(data)*val_perc)]
    train_data, val_data = random_split(data, lengths)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory=True)

    num_batches = len(train_data) // batch_size
    if len(train_data) % batch_size != 0:
        num_batches += 1

    print(f'\nnum train samples = {len(train_data)} --> {num_batches} train batches')
    print(f'num val samples = {len(val_data)}')

    model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="min", monitor='Validation/Loss', save_last=True)
    trainer=pl.Trainer(
                    precision=32, #32 is normal, 16 is mixed precision
                    progress_bar_refresh_rate=1, #every N batches update progress bar
                    log_every_n_steps=10,
                    callbacks=[model_checkpoint],
                    gpus=torch.cuda.device_count(),
                    accelerator='dp', #anything else here seems to lead to crashes/errors
                    default_root_dir=log_dir,
                    val_check_interval=val_check_interval if val_check_interval > 1 else float(val_check_interval),
                    max_epochs=epochs
                )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--margin', default=0.8, type=float)
    parser.add_argument('--n', default=10, type=int, help='Horizon for n-step TD error')
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--version', default=0, type=int, help='Version of model, if training is resumed from checkpoint')
    parser.add_argument('--vqvae_path', help='Path to encoding model')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data/numpy_data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--num_data', default=0, type=int, help='Number of datapoints to use')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr_gamma', default=1, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--val_perc', default=0.1, type=float, help='How much of the data should be used for validation')
    parser.add_argument('--val_check_interval', default=1, type=int, help='How often to validate. N == 1 --> once per epoch; N > 1 --> every N steps')
    parser.add_argument('--target_update_rate', default=1, type=int, help='How often to update target network')
    parser.add_argument('--centroids_path', type=str, default='/home/lieberummaas/datadisk/minerl/data/numpy_data')
    
    args = parser.parse_args()
    
    main(**vars(args))