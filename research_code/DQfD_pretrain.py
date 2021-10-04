import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from einops.layers.torch import Rearrange

import numpy as np
import os
import argparse
import einops
from copy import deepcopy

from vqvae import VQVAE
from vae_model import VAE

import datasets

class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = nn.functional.relu(out)
        return out

class ConvFeatureExtractor(nn.Module):
    def __init__(self, n_hid=86, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4)
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
        target_update_rate, # how often to update the target network
        margin, # margin in the classification loss
        discount_factor, 
        horizon, # time horizon for the N-step TD error
        feature_extractor_cls=VQVAE,
        feature_extractor_path=None,
        feature_extractor_kwargs={},
        freeze_feature_extractor=False
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
            nn.Conv2d(in_channels=num_channels, out_channels=128, kernel_size=3, padding=1, stride=1), # 16 -> 8
            nn.GELU(),
            nn.AdaptiveMaxPool2d((8,8)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1), # 8 -> 4
            nn.GELU(),
            nn.AdaptiveMaxPool2d((4,4)),
            #nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=2), # 4 -> 2
            nn.AdaptiveMaxPool2d((2,2)),
            Rearrange('b c h w -> b (c h w)')
        )
        dummy = self.conv_net(dummy)
        pov_feature_dim = dummy.shape[1]

        self.vecobs_featurizer = nn.Sequential(
            nn.Linear(64, 100),
            nn.GELU(),
            nn.Linear(100, 100)
        )

        self.q_net = nn.Sequential(
            nn.Linear(100 + pov_feature_dim, 150),
            nn.GELU(),
            nn.Linear(150, self.hparams.n_actions)
        )
        
        # init target net
        self._update_target()
        
        # loss function
        self.loss_fn = nn.MSELoss()
    
    def _update_target(self):
        self.target_net = nn.ModuleDict({
            'feature_extractor':deepcopy(self.feature_extractor),
            'conv_net':deepcopy(self.conv_net),
            'vecobs_featurizer':deepcopy(self.vecobs_featurizer),
            'q_net':deepcopy(self.q_net),
        })
        self.target_net.eval()
        
    def forward(self, pov, vec_obs, target=False):
        if target:
            # extract pov features
            pov_out, _, _ = self.target_net['feature_extractor'].encode_only(pov)
            
            # apply conv net
            pov_out = self.target_net['conv_net'](pov_out)
            
            # extract vec obs features
            vec_out = self.target_net['vecobs_featurizer'](vec_obs)
            
            # compute q_values
            q_values = self.target_net['q_net'](torch.cat([pov_out, vec_out], dim=1))
            
            return q_values
        else:
            # extract pov features
            if self.hparams.freeze_feature_extractor:
                pov_out, *_ = self.feature_extractor.encode_only(pov)
                codebook_loss = 0
            else:
                pov_out, codebook_loss, _, _ = self.feature_extractor.encode_with_grad(pov)
            
            # apply conv net
            pov_out = self.conv_net(pov_out)
            
            # extract vec obs features
            vec_out = self.vecobs_featurizer(vec_obs)
            
            # compute q_values
            q_values = self.q_net(torch.cat([pov_out, vec_out], dim=1))

            return q_values, codebook_loss
    
    def _large_margin_classification_loss(self, q_values, expert_action):
        '''
        Computes the large margin classification loss J_E(Q) from the DQfD paper
        '''
        idcs = torch.arange(0,len(q_values),dtype=torch.long)
        q_values = q_values + self.hparams.margin
        q_values[idcs, expert_action] = q_values[idcs, expert_action] - self.hparams.margin
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
        expert_q_values = q_values[idcs, action].mean()
        other_q_values =  deepcopy(q_values.detach())
        other_q_values[idcs, action] = 0
        other_q_values = other_q_values.mean()
        classification_loss = self._large_margin_classification_loss(q_values, action).mean()
        one_step_loss = self.loss_fn(q_values[idcs, action], reward + self.hparams.discount_factor * target_next_q_values[idcs, base_next_action])
        n_step_loss = self.loss_fn(q_values[idcs, action], n_step_reward + (self.hparams.discount_factor ** self.hparams.horizon) * target_n_step_q_values[idcs, base_n_step_action])

        # sum up losses
        loss = classification_loss + one_step_loss + n_step_loss + codebook_loss

        # compute perc where expert action has highest q_value for logging
        expert_agent_agreement = (torch.argmax(q_values, dim=1) == action).sum() / q_values.shape[0]
        
        # logging
        self.log('Training/1-step TD Error', one_step_loss, on_step=True)
        self.log('Training/ClassificationLoss', classification_loss, on_step=True)
        self.log('Training/n-step TD Error', n_step_loss, on_step=True)
        self.log('Training/Loss', loss, on_step=True)
        self.log('Training/ExpertAgentAgreement', expert_agent_agreement, on_step=True)
        self.log('Training/ExpertQValues', expert_q_values, on_step=True)
        self.log('Training/OtherQValues', other_q_values, on_step=True)
        self.logger.experiment.add_histogram('Training/Actions', action, global_step=self.global_step)
        
        return loss
    
    def on_after_backward(self):
        if (self.global_step + 1) % self.hparams.target_update_rate == 0:
            print(f'\nGlobal step {self.global_step+1}: Updating Target Network\n')
            self._update_target()
        
    def configure_optimizers(self):
        # set up optimizer
        params = list(self.conv_net.parameters()) + list(self.vecobs_featurizer.parameters()) + list(self.q_net.parameters())
        if not self.hparams.freeze_feature_extractor:
            params += list(self.feature_extractor.parameters())
        optimizer = torch.optim.AdamW(params, **self.hparams.optim_kwargs)
        return optimizer


def main(
    env_name, 
    batch_size, 
    num_workers, 
    lr, 
    weight_decay, 
    feature_extractor_path, 
    data_dir, 
    log_dir,
    epochs, 
    feature_extractor_cls, 
    freeze_feature_extractor,
    centroids_path, 
    target_update_rate, 
    margin, 
    discount_factor, 
    horizon
):
    pl.seed_everything(1337)


    # load centroids
    centroids_path = os.path.join(centroids_path, env_name + '_150_centroids.npy')
    print(f'\nLoading centroids from {centroids_path}...')
    centroids = np.load(centroids_path)
    print(f'Loaded centroids! Shape is {centroids.shape}.')


    if feature_extractor_cls == 'conv' and freeze_feature_extractor:
        raise ValueError("Mustn't freeze_feature_extractor when using conv!")
    
    ## some model kwargs
    optim_kwargs = {'lr':lr, 'weight_decay':weight_decay}
    feature_extractor_cls = {
        'vqvae':VQVAE,
        'vae':VAE,
        'conv':ConvFeatureExtractor
    }[feature_extractor_cls]
    model_kwargs = {
        'feature_extractor_path':feature_extractor_path,
        'optim_kwargs':optim_kwargs,
        'n_actions':centroids.shape[0],
        'target_update_rate':target_update_rate,
        'margin':margin,
        'discount_factor':discount_factor,
        'horizon':horizon,
        'feature_extractor_cls':feature_extractor_cls,
        'freeze_feature_extractor':freeze_feature_extractor
    }
    
    # make sure that relevant dirs exist
    run_name = f'QNetwork/{env_name}/{feature_extractor_cls.__name__}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    model = QNetwork(**model_kwargs)
        
    
    # load data
    train_data = datasets.PretrainQNetIterableData(env_name, data_dir, centroids, horizon, discount_factor, num_workers)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    model_checkpoint = ModelCheckpoint(mode="min", monitor='Training/Loss', save_last=True, every_n_train_steps=500)
    trainer=pl.Trainer(
                    progress_bar_refresh_rate=1, #every N batches update progress bar
                    log_every_n_steps=10,
                    callbacks=[model_checkpoint],
                    gpus=torch.cuda.device_count(),
                    #accelerator='dp', #anything else here seems to lead to crashes/errors
                    default_root_dir=log_dir,
                    max_epochs=epochs
                )
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLNavigateDenseVectorObf-v0')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--margin', default=0.8, type=float)
    parser.add_argument('--horizon', default=50, type=int, help='Horizon for n-step TD error')
    parser.add_argument('--feature_extractor_cls', choices=['vqvae', 'vae', 'conv'], default='vqvae', help='Class of the feature_extractor model')
    parser.add_argument('--feature_extractor_path', help='Path to feature_extractor model')
    parser.add_argument('--freeze_feature_extractor', action='store_true', help='Whether to freeze or finetune the feature extractor')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--target_update_rate', default=100, type=int, help='How often to update target network')
    parser.add_argument('--centroids_path', type=str, default='/home/lieberummaas/datadisk/minerl/data/')
    
    args = parser.parse_args()
    
    main(**vars(args))