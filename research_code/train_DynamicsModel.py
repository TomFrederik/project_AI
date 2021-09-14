import dynamics_models
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
import einops


# for debugging
torch.autograd.set_detect_anomaly(True)

STR_TO_MODEL = {
    'rssm':dynamics_models.RSSM,
    'mdn':dynamics_models.MDN_RNN
}

class PredictionCallback(pl.Callback):

    def __init__(self, every_n_epochs=1, dataset=None, every_n_batches=100, seq_len=10):
        """
        Inputs:
            every_n_epochs - Only save those images every N epochs
            dataset - Dataset to sample from
            save_to_disk - If True, the samples and image means should be saved to disk as well.
            every_n_batches - Only save those images every N batches
            seq_len - maximum seq len of the sample, if dataset only contains shorter samples then that is the maximum seq len instead
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches

        #x_samples, x_mean = pl_module.sample(self.batch_size)
        #pov, vec_obs, act = map(lambda x: x[None,:seq_len], next(iter(dataset))[:-1])
        pov, vec_obs, act = map(lambda x: x[:,:seq_len], dataset[0][:-1])
        pov = torch.from_numpy(pov)
        vec = torch.from_numpy(vec_obs)
        act = torch.from_numpy(act)
        self.seq_len = pov.shape[1]
        self.sequence = (pov, vec, act)

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.predict_sequence(trainer, pl_module, trainer.current_epoch+1)
    
    def on_batch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (pl_module.global_step+1) % self.every_n_batches == 0:
            self.predict_sequence(trainer, pl_module, pl_module.global_step+1)

    def predict_sequence(self, trainer, pl_module: dynamics_models.MDN_RNN, epoch):
        """
        Function that predicts sequence and generates images.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        # make sure sequence is on correct device
        if self.sequence[0].device != pl_module.device:
            self.sequence = list(map(lambda x: x.to(pl_module.device), self.sequence))
        
        # predict sequence
        _, pov_samples, _ = pl_module.forward(*self.sequence)
        if pl_module.hparams.VAE_class == 'vqvae':
            pov_samples = einops.rearrange(pov_samples, 'b t c (h w) -> (b t) c h w', h=16, w=16)
            
            # reconstruct images
            pov_reconstruction = pl_module.VAE.decode_only(pov_samples)
            
            # stack images
            images = torch.stack([self.sequence[0][0,1:], pov_reconstruction], dim=1).reshape(((self.seq_len -1) * 2, 3, 64, 64))

            # log images to tensorboard
            trainer.logger.experiment.add_image('Prediction', make_grid(images, nrow=2), epoch)
        
        else:
            pov_samples = einops.rearrange(pov_samples, 'b t c h w -> (b t h w) c')
            # reconstruct images
            pov_reconstruction = pl_module.VAE.decode_only(pov_samples)

            images = torch.stack([self.sequence[0][0,1:], pov_reconstruction], dim=1).reshape(((self.seq_len -1) * 2, 3, 64, 64))

            # log images to tensorboard
            trainer.logger.experiment.add_image('Prediction', make_grid(images, nrow=2), epoch)


def train_DynamicsModel(env_name, data_dir, dynamics_model, seq_len, lr, 
                        val_perc, batch_size, num_data, epochs, 
                        lr_gamma, lr_decrease_freq, log_dir, lr_step_mode, 
                        model_path, VAE_class, vae_version, num_components,
                        val_check_interval, load_from_checkpoint, version,
                        profile, temp,
                        conditioning_len, curriculum_threshold, curriculum_start,
                        save_freq):
    
    pl.seed_everything(1337)

    if VAE_class == 'vae':
        vae_path = os.path.join(log_dir, 'VAE', env_name, 'lightning_logs', 'version_'+str(vae_version), 'checkpoints/last.ckpt')
    elif VAE_class == 'vqvae':
        #vae_path = os.path.join(log_dir, 'VQVAE', env_name, 'lightning_logs', 'version_'+str(vae_version), 'checkpoints/last.ckpt')
        vae_path = os.path.join(log_dir, 'VQVAE', env_name, 'lightning_logs', 'version_'+str(vae_version), 'checkpoints/last.ckpt')

    # make sure that relevant dirs exist
    run_name = f'DynamicsModel/{STR_TO_MODEL[dynamics_model].__name__}/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    ## some model kwargs
    optim_kwargs = {'lr':lr}
    scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
    
    if dynamics_model == 'rssm':
        raise NotImplementedError
        """
        seq_len = seq_len        
        lstm_kwargs = {'num_layers':1, 'hidden_size':2048}
        model_kwargs = {
            'lstm_kwargs':lstm_kwargs, 
            'seq_len':seq_len, 
            'VAE_path':model_path,
            'optim_kwargs':optim_kwargs,
            'scheduler_kwargs':scheduler_kwargs,
            'VAE_class':VAE_class,
            'latent_overshooting':latent_overshooting
        }
        monitor = 'Validation/loss'
        """
    elif dynamics_model == 'mdn':
        gru_kwargs = {'num_layers':1, 'hidden_size':16*16*32}
        model_kwargs = {
            'gru_kwargs':gru_kwargs, 
            'seq_len':seq_len, 
            'VAE_path':model_path,
            'optim_kwargs':optim_kwargs,
            'scheduler_kwargs':scheduler_kwargs,
            'VAE_class':VAE_class,
            'VAE_path':vae_path,
            'num_components':num_components,
            'temp':temp,
            'conditioning_len':conditioning_len,
            'curriculum_threshold':curriculum_threshold,
            'curriculum_start':curriculum_start,
        }
        monitor = 'Training/loss'
    else:
        ValueError(f"Unrecognized model {dynamics_model}")
    ##
    
    # init model
    if load_from_checkpoint:
        checkpoint = os.path.join(log_dir, 'lightning_logs', 'version_'+str(version), 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint}')
        model = STR_TO_MODEL[dynamics_model].load_from_checkpoint(checkpoint)
    else:
        model = STR_TO_MODEL[dynamics_model](**model_kwargs)

    # load data
    #train_data = datasets.SingleSequenceDynamics(env_name, data_dir, seq_len + conditioning_len, batch_size)
    train_data = datasets.DynamicsData(env_name, data_dir, seq_len + conditioning_len, batch_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, pin_memory=True)

    model_checkpoint = ModelCheckpoint(mode="min", monitor=monitor, save_last=True, every_n_train_steps=save_freq)
    '''
    prediction_callback = PredictionCallback(
        every_n_batches=save_freq,
        dataset=train_data,
        seq_len=10
    )'''
    #callbacks = [model_checkpoint, prediction_callback]
    callbacks = [model_checkpoint]
    trainer=pl.Trainer(
        progress_bar_refresh_rate=1, #every N batches update progress bar
        log_every_n_steps=1,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        accelerator='dp', #anything else here seems to lead to crashes/errors
        default_root_dir=log_dir,
        max_epochs=epochs,
        track_grad_norm=2,
    )
    trainer.fit(model, train_loader)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', help='Path to encoding model')
    parser.add_argument('--data_dir', default="/home/lieberummaas/datadisk/minerl/data")
    parser.add_argument('--log_dir', default="/home/lieberummaas/datadisk/minerl/run_logs")
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--dynamics_model', default='mdn', choices=['rssm', 'mdn'], help='Model used to predict the next latent state')
    parser.add_argument('--seq_len', default=4, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_data', default=0, type=int, help='Number of datapoints to use')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--save_freq', default=100, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_gamma', default=1, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--val_perc', default=0.1, type=float, help='How much of the data should be used for validation')
    parser.add_argument('--VAE_class', type=str, default='vae', choices=['vae', 'vqvae'])
    parser.add_argument('--vae_version', type=int, default=0)
    parser.add_argument('--num_components', type=int, default=5, help='Number of mixture components. Only used in MDN-RNN')
    parser.add_argument('--val_check_interval', default=1, type=int, help='How often to validate. N == 1 --> once per epoch; N > 1 --> every N steps')
    parser.add_argument('--load_from_checkpoint', action='store_true')
    #parser.add_argument('--latent_overshooting', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--temp', default=1, type=float)
    parser.add_argument('--curriculum_threshold', default=3, type=float)
    parser.add_argument('--curriculum_start', default=0, type=int)
    parser.add_argument('--conditioning_len', default=0, type=int, help='Length of sequence to condition rnn on')
    parser.add_argument('--version', default=0, type=int, help='Version directory of model, if training is resumed from checkpoint')

    args = vars(parser.parse_args())

    train_DynamicsModel(**args)
