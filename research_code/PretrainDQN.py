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

from vqvae import VQVAE
from visual_models import VAE

import datasets

#torch.backends.cudnn.benchmark = True

class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), # 64 -> 32
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), # 32 -> 16
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, 1) # 16 -> 16
        )
    def forward(self, x):
        return self.conv(x)
    
    @torch.no_grad()
    def encode_only(self, x):
        return self(x), None, None
    
    def encode_with_grad(self, x):
        return self(x), 0, None, None
    
    @property
    def device(self):
        return list(self.conv.parameters())[0].device

class QNetwork(pl.LightningModule):
    
    def __init__(
        self, 
        n_actions, # number of distinct actions
        optim_kwargs, 
        scheduler_kwargs, 
        target_update_rate, # how often to update the target network
        margin, # margin in the classification loss
        discount_factor, 
        n, # time horizon for the N-step TD error
        feature_extractor_cls=VQVAE,
        feature_extractor_path=None,
        feature_extractor_kwargs={},
        train_feature_extractor=False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # set up feature extractor
        if feature_extractor_cls in [VQVAE, VAE]:
            self.feature_extractor = feature_extractor_cls.load_from_checkpoint(feature_extractor_path)
        elif feature_extractor_cls == ConvFeatureExtractor:
            self.feature_extractor = feature_extractor_cls(**feature_extractor_kwargs)
        else:
            raise ValueError(f'Unrecognized feature extractor class {feature_extractor_cls}')

        # conv feature extractor
        dummy, dummy_idcs, _ = self.feature_extractor.encode_only(torch.ones(2,3,64,64).float().to(self.feature_extractor.device))
        num_channels = dummy.shape[1]
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=256, kernel_size=3, padding=1, stride=2), # 16 -> 8
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2), # 8 -> 4
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2), # 4 -> 2
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=self.hparams.n_actions, kernel_size=3, padding=1, stride=2)#, # 2 -> 1
        )
        
        # init target net
        self._update_target()
        
        # loss function
        self.loss_fn = nn.MSELoss()
    
    def _update_target(self):
        self.target_net = nn.ModuleDict({
            'feature_extractor':deepcopy(self.feature_extractor),
            'conv_net':deepcopy(self.conv_net),
        })
        self.target_net.eval()
        
    def forward(self, pov, vec_obs, target=False):
        if target:
            # extract features
            out, _, _ = self.target_net['feature_extractor'].encode_only(pov)
            # apply conv net
            out = einops.rearrange(self.target_net['conv_net'](out), 'b c h w -> b (c h w)')
            return out
        else:
            # extract features
            if self.hparams.train_feature_extractor:
                out, codebook_loss, _, _ = self.feature_extractor.encode_with_grad(pov)
            else:
                out, *_ = self.feature_extractor.encode_only(pov)
                codebook_loss = 0

            # apply conv net
            out = einops.rearrange(self.conv_net(out), 'b c h w -> b (c h w)')
            return out, codebook_loss
    
    def _large_margin_classification_loss(self, q_values, expert_action):
        '''
        Computes the large margin classification loss J_E(Q) from the DQfD paper
        '''
        idcs = torch.arange(0,len(q_values),dtype=torch.long)
        q_values = q_values + self.hparams.margin
        q_values[idcs, expert_action] = q_values[idcs,expert_action] - self.hparams.margin
        return torch.max(q_values, dim=1)[0] - q_values[idcs,expert_action]
    
    def training_step(self, batch, batch_idx):
        pov, vec_obs, action, reward, next_pov, next_vec_obs, n_step_reward, n_step_pov, n_step_vec_obs = batch
        
        # predict q values
        q_values, codebook_loss = self(pov, vec_obs)
        action = action.detach()
        target_next_q_values = self(next_pov, next_vec_obs, target=True).detach()
        base_next_action = torch.argmax(self(next_pov, next_vec_obs)[0].detach(), dim=1)
        target_n_step_q_values = self(n_step_pov, n_step_vec_obs, target=True).detach()
        base_n_step_action = torch.argmax(self(n_step_pov, n_step_vec_obs)[0].detach(), dim=1)
        
        # compute the individual losses
        idcs = torch.arange(0, len(q_values), dtype=torch.long, requires_grad=False)
        classification_loss = self._large_margin_classification_loss(q_values, action).mean()
        one_step_loss = self.loss_fn(q_values[idcs, action], reward + self.hparams.discount_factor * target_next_q_values[idcs, base_next_action])
        n_step_loss = self.loss_fn(q_values[idcs, action], n_step_reward + (self.hparams.discount_factor ** self.hparams.n) * target_n_step_q_values[idcs, base_n_step_action])

        # sum up losses
        loss = classification_loss + one_step_loss + n_step_loss + codebook_loss
        
        # logging
        self.log('Training/1-step TD Error', one_step_loss, on_step=True)
        self.log('Training/n-step TD Error', n_step_loss, on_step=True)
        self.log('Training/Loss', loss, on_step=True)
        
        return loss
    
    def on_after_backward(self):
        if (self.global_step + 1) % self.hparams.target_update_rate == 0:
            print(f'\nGlobal step {self.global_step+1}: Updating Target Network\n')
            self._update_target()
        
    def configure_optimizers(self):
        # set up optimizer
        params = list(self.conv_net.parameters())
        if self.hparams.train_feature_extractor:
            params += list(self.feature_extractor.parameters())
        optimizer = torch.optim.AdamW(params, **self.hparams.optim_kwargs)
        # set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.hparams.scheduler_kwargs['lr_step_mode'],
            'frequency': self.hparams.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}


def main(
    env_name, batch_size, num_workers, lr, weight_decay, load_from_checkpoint, 
    version, feature_extractor_path, data_dir, log_dir,
    num_data, epochs, lr_gamma, lr_step_mode, lr_decrease_freq, feature_extractor_cls, train_feature_extractor,
    centroids_path, target_update_rate, margin, discount_factor, n
):
    pl.seed_everything(1337)


    # load centroids
    centroids_path = os.path.join(centroids_path, env_name + '_150_centroids.npy')
    print(f'\nLoading centroids from {centroids_path}...')
    centroids = np.load(centroids_path)
    print(f'Loaded centroids! Shape is {centroids.shape}.')
    
    ## some model kwargs
    optim_kwargs = {'lr':lr, 'weight_decay':weight_decay}
    scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
    feature_extractor_cls = {
        'vqvae':VQVAE,
        'vae':VAE,
        'conv':ConvFeatureExtractor
    }[feature_extractor_cls]
    model_kwargs = {
        'feature_extractor_path':feature_extractor_path,
        'optim_kwargs':optim_kwargs,
        'scheduler_kwargs':scheduler_kwargs,
        'n_actions':centroids.shape[0],
        'target_update_rate':target_update_rate,
        'margin':margin,
        'discount_factor':discount_factor,
        'n':n,
        'feature_extractor_cls':feature_extractor_cls,
        'train_feature_extractor':train_feature_extractor
    }
    
    # make sure that relevant dirs exist
    run_name = f'QNetwork/{env_name}/{feature_extractor_cls.__name__}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    if load_from_checkpoint:
        checkpoint_file = os.path.join(log_dir, 'lightning_logs', f'version_{version}', 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint_file}')
        model = QNetwork.load_from_checkpoint(checkpoint_file, **model_kwargs)
    else:
        model = QNetwork(**model_kwargs)
        
    
    # load data
    train_data = datasets.PretrainQNetIterableData(env_name, data_dir, centroids, n, discount_factor, num_workers)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="min", monitor='Training/Loss', save_last=True, every_n_train_steps=500)
    trainer=pl.Trainer(
                    precision=32, #32 is normal, 16 is mixed precision
                    progress_bar_refresh_rate=1, #every N batches update progress bar
                    log_every_n_steps=10,
                    callbacks=[model_checkpoint],
                    gpus=torch.cuda.device_count(),
                    accelerator='dp', #anything else here seems to lead to crashes/errors
                    default_root_dir=log_dir,
                    max_epochs=epochs
                )
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--margin', default=0.8, type=float)
    parser.add_argument('--n', default=50, type=int, help='Horizon for n-step TD error')
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--version', default=0, type=int, help='Version of model, if training is resumed from checkpoint')
    parser.add_argument('--feature_extractor_cls', choices=['vqvae', 'vae', 'conv'], default='vqvae', help='Class of the feature_extractor model')
    parser.add_argument('--feature_extractor_path', help='Path to feature_extractor model')
    parser.add_argument('--train_feature_extractor', action='store_true', help='Whether to train or finetune the feature extractor')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--num_data', default=0, type=int, help='Number of datapoints to use')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_gamma', default=1, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--target_update_rate', default=100, type=int, help='How often to update target network')
    parser.add_argument('--centroids_path', type=str, default='/home/lieberummaas/datadisk/minerl/data/')
    
    args = parser.parse_args()
    
    main(**vars(args))