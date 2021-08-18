import argparse
import os
import datasets
import visual_models
import vqvae


from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torch.optim import AdamW

import einops
import numpy as np

vae_model_by_str = {
    'Conv':visual_models.ConvVAE,
    'ResNet':visual_models.ResnetVAE,
    'vqvae':vqvae.VQVAE
}

class InventoryPredictor(pl.LightningModule):
    def __init__(self, optim_kwargs, scheduler_kwargs, VAE_path, 
                 VAE_class='vqvae'):
        super().__init__()
        self.save_hyperparameters()
        
        # load VAE
        self.VAE = vae_model_by_str[VAE_class].load_from_checkpoint(VAE_path)
    
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        dummy, dummy_idcs, _ = self.VAE.encode_only(torch.ones(2,3,64,64).float().to(self.VAE.device))
        if self.hparams.predict_idcs_directly and self.hparams.embed:
            dummy = einops.rearrange(self.embedding(dummy_idcs), 'b h w D -> b D h w')
        self.latent_h = dummy.shape[-1]
        self.latent_size = np.prod(dummy.shape[2:])
            
        num_channels = dummy.shape[1]
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=256, kernel_size=3, padding=1, stride=2), # 16 -> 8
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2), # 8 -> 4
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2), # 4 -> 2
            nn.GELU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, stride=2)#, # 2 -> 1
            #nn.GELU()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, pov, vec_obs, act):
        out, _, _ = self.VAE.encode_only(pov)
        out = self.conv_net(out)
        out = torch.cat([out, vec_obs, act], dim=1)
        out = self.mlp(out)
    
    def configure_optimizers(self):
        # set up optimizer
        optimizer =  AdamW(self.parameters(), **self.hparams.optim_kwargs)
        # set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}
    
    def training_step(self, batch, batch_idx):
        pov, vec_obs, act, targets = batch
        pred = self(pov, vec_obs, act)
        loss = self.loss_fn(pred, targets)
        accuracy = (targets[torch.sigmoid(pred) >= 0.5].sum() + (1 - targets)[torch.sigmoid(pred) < 0.5].sum()) / len(targets) 
        self.log('Training/Accuracy', accuracy, on_step=True)
        self.log('Training/Loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pov, vec_obs, act, targets = batch
        pred = self(pov, vec_obs, act)
        loss = self.loss_fn(pred, targets)
        accuracy = (targets[torch.sigmoid(pred) >= 0.5].sum() + (1 - targets)[torch.sigmoid(pred) < 0.5].sum()) / len(targets) 
        self.log('Validation/Accuracy', accuracy, on_epoch=True)
        self.log('Validation/Loss', loss, on_epoch=True)
        return loss

def main(env_name, batch_size, lr, load_from_checkpoint, version, vae_path, data_dir, log_dir,
        num_data, epochs, lr_gamma, lr_step_mode, lr_decrease_freq, val_perc, val_check_interval, VAE_class):
    # make sure that relevant dirs exist
    run_name = f'PredictInventory/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    ## some model kwargs
    optim_kwargs = {'lr':lr}
    scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
    model_kwargs = {
        'VAE_path':vae_path,
        'optim_kwargs':optim_kwargs,
        'scheduler_kwargs':scheduler_kwargs,
        'VAE_class':VAE_class
    }

    if load_from_checkpoint:
        checkpoint_file = os.path.join(log_dir, 'lightning_logs', f'version_{version}', 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint_file}')
        model = InventoryPredictor.load_from_checkpoint(checkpoint_file, **model_kwargs)
    else:
        model = InventoryPredictor(**model_kwargs)
        
    
    # load data
    data = datasets.VectorObsData(env_name, data_dir, num_data)
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
    p = argparse.ArgumentParser()
    p.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    p.add_argument('--batch_size', default=100)
    p.add_argument('--lr', default=3e-4)
    p.add_argument('--load_from_checkpoint', action='store_true')
    p.add_argument('--version', default=0, type=int, help='Version of model, if training is resumed from checkpoint')
    p.add_argument('--model_path', help='Path to encoding model')
    p.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data/numpy_data')
    p.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    p.add_argument('--num_data', default=0, type=int, help='Number of datapoints to use')
    p.add_argument('--epochs', default=1, type=int)
    p.add_argument('--lr_gamma', default=1, type=float, help='Learning rate adjustment factor')
    p.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    p.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    p.add_argument('--val_perc', default=0.1, type=float, help='How much of the data should be used for validation')
    p.add_argument('--val_check_interval', default=1, type=int, help='How often to validate. N == 1 --> once per epoch; N > 1 --> every N steps')
    p.add_argument('--VAE_class', default='vqvae')    
    
    args = p.parse_args()
    
    main(vars(args))