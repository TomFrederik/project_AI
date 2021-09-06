
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import os
import argparse

import datasets
from vecobs_vqvae import VecObsVQVAE


class RewardMLP(pl.LightningModule):
    def __init__(self, hidden_dims, learning_rate, scheduler_kwargs, quantizer_path):
        super().__init__()
        self.save_hyperparameters()

        # set up quantizer
        if quantizer_path is not None:
            self.use_quantizer = True
            print(f'\nLoading quantizer from {quantizer_path}')
            self.quantizer = VecObsVQVAE.load_from_checkpoint(quantizer_path)
            self.quantizer.eval()
            dummy_vec = torch.ones(1,64).to(self.quantizer.device)
            dummy_quant = self.quantizer.encode_only(dummy_vec)[0]
            self.input_dim = dummy_quant.shape[-1]
        else:
            self.use_quantizer = False
            self.input_dim = 64

        # create MLP
        self.mlp = [nn.Sequential(nn.Linear(self.input_dim, hidden_dims[0]), nn.GELU())]
        for i in range(len(hidden_dims)-1):
            self.mlp.append(nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.GELU()))
        self.mlp.append(nn.Linear(hidden_dims[-1], 1))
        self.mlp = nn.Sequential(*self.mlp)
        
        # set up loss function
        self.loss_fn = nn.MSELoss(reduction='none')
        

    def forward(self, vec_obs):
        out = vec_obs
        if self.use_quantizer:
            out = self.quantizer.encode_only(out)[0]
        return self.mlp(out)
    
    def training_step(self, batch, batch_idx):
        obs, _, reward, *_ = batch
        vec_obs = obs['vector'][0].float()
        reward = torch.log2(1+reward[0])
        loss_scaling_factor = 2 ** reward.detach()
        predicted_reward = self.forward(vec_obs)[:,0]
        
        loss = (self.loss_fn(predicted_reward, reward) * loss_scaling_factor).mean()

        self.log('Training/loss', loss, on_step=True)
        return loss

    def configure_optimizers(self):
        # set up optimizer, only train mlp params
        optimizer =  AdamW(self.mlp.parameters(), lr=self.hparams.learning_rate)
        # set up 
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.hparams.scheduler_kwargs['lr_step_mode'],
            'frequency': self.hparams.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}

def train(
    env_name, 
    data_dir, 
    log_dir, 
    batch_size,
    epochs, 
    save_freq,
    lr,
    lr_gamma, 
    lr_decrease_freq, 
    lr_step_mode,
    model_class, 
    load_from_checkpoint, 
    version,
    quantizer_version
):
    pl.seed_everything(1337)

    # set quantizer_path
    if quantizer_version is None:
        quantizer_path = None
    else:
        quantizer_path = os.path.join(log_dir, 'VecObsVQVAE', env_name, 'lightning_logs', 'version_'+str(quantizer_version), 'checkpoints', 'last.ckpt')

    # make sure that relevant dirs exist
    run_name = f'RewardModel_{model_class}/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'Saving logs and model to {log_dir}')

    # set hidden dims and input dim
    # TODO parse these as args
    hidden_dims = [100,100]
    
    # instantiate model
    if load_from_checkpoint:
        checkpoint = os.path.join(log_dir, 'lightning_logs', 'version_'+str(version), 'checkpoints', 'last.ckpt')        
        print(f'Loading model from {checkpoint}')
        model = RewardMLP.load_from_checkpoint(checkpoint, lr=lr)
    else:
        scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
        model = RewardMLP(hidden_dims, lr, scheduler_kwargs, quantizer_path)
        
    # load data
    data = datasets.BufferedBatchDataset(env_name, data_dir, batch_size, epochs)
    dataloader = DataLoader(data)

    # create callbacks to sample reconstructed images and for model checkpointing
    checkpoint_callback = ModelCheckpoint(mode="min", monitor="Training/loss", save_last=True, every_n_train_steps=save_freq)
    trainer=pl.Trainer(
        progress_bar_refresh_rate=1, #every N batches update progress bar
        callbacks=[checkpoint_callback],
        gpus=torch.cuda.device_count(),
        default_root_dir=log_dir,
        max_epochs=epochs,
        log_every_n_steps=10,
        track_grad_norm=2
    )
                    
    # fit model
    trainer.fit(model, dataloader)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--env_name', default='MineRLObtainIronPickaxeVectorObf-v0')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--save_freq', default=100, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--load_from_checkpoint', default=False, action='store_true')
    parser.add_argument('--version', default=0, type=int, help='Version of model, if training is resumed from checkpoint')
    parser.add_argument('--model_class', default='MLP', type=str)
    parser.add_argument('--quantizer_version', default=None, type=int)

    args = vars(parser.parse_args())

    train(**args)
