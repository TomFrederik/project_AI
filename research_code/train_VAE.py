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

STR_TO_CLASS = {'Conv':models.ConvVAE, 'CLSTM':models.CLSTMVAE}

class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=6, every_n_epochs=1, dataset=None, save_to_disk=False, every_n_batches=100):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            dataset - Dataset to sample from
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches
        self.save_to_disk = save_to_disk
        self.dataset = dataset

        #x_samples, x_mean = pl_module.sample(self.batch_size)
        idx = np.random.choice(len(self.dataset), replace=False, size=self.batch_size)
        batch = [self.dataset[i] for i in idx]
        self.img_batch = torch.stack([b[1] for b in batch], dim=0)

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)
    
    def on_batch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (pl_module.global_step+1) % self.every_n_batches == 0:
            self.sample_and_save(trainer, pl_module, pl_module.global_step+1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        # Hints:
        # - You can access the logging directory path via trainer.logger.log_dir, and
        # - You can access the tensorboard logger via trainer.logger.experiment
        # - Use the torchvision function "make_grid" to create a grid of multiple images
        # - Use the torchvision function "save_image" to save an image grid to disk 

        #x_samples, x_mean = pl_module.sample(self.batch_size)
        
        if self.img_batch.device != pl_module.device:
            self.img_batch = self.img_batch.to(pl_module.device)

        reconstructed_img = pl_module.reconstruct_only(self.img_batch)

        images = torch.stack([self.img_batch, reconstructed_img], dim=1).reshape((self.batch_size * 2, *self.img_batch.shape[1:]))

        # log images to tensorboard
        trainer.logger.experiment.add_image('Reconstruction',make_grid(images, nrow=2), epoch)


def train_VAE(env_name, data_dir, lr, val_perc, eval_freq, batch_size, epochs, lr_gamma, lr_decrease_freq, log_dir, model_class, lr_step_mode):
    
    # make sure that relevant dirs exist
    run_name = f'{model_class}_VAE/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'Saving logs and model to {log_dir}')

    kernel_size = 3
    num_frames = 1
    img_shape = (64,64)
    latent_dim = 128 # CHANGED
    num_encoder_channels = [32,64,128,256] # channels for encoder
    num_decoder_channels = num_encoder_channels.copy()
    num_decoder_channels.reverse()
    print(f'\nnum_encoder_channels = {num_encoder_channels}')
    print(f'num_decoder_channels = {num_decoder_channels}')

    '''
    # generate lstm kwarg list
    lstm_kwarg_list = [{'out_channels':n} for n in num_encoder_channels]
    lstm_kwarg_list[0]['in_channels'] = 3
    lstm_kwarg_list[0]['img_shape'] = (64,64)
    lstm_kwarg_list[0]['kernel_size'] = kernel_size
    for i in range(1, len(lstm_kwarg_list)):
        lstm_kwarg_list[i]['in_channels'] = num_encoder_channels[i-1]
        lstm_kwarg_list[i]['kernel_size'] = kernel_size
    
    # set encoder and decoder kwargs
    encoder_kwargs = {
        'lstm_kwarg_list':lstm_kwarg_list,
        'num_frames':num_frames,
        'latent_dim':latent_dim
    }
    '''
    
    encoder_kwargs = {
        'img_shape':img_shape,
        'latent_dim':latent_dim,
        'num_channels':num_encoder_channels,
        'kernel_size':kernel_size
    }
    decoder_kwargs = {
        'latent_dim':latent_dim,
        'num_channels':num_decoder_channels,
        'kernel_size':kernel_size
    }
    
    
    # init model
    scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
    model = STR_TO_CLASS[model_class](encoder_kwargs, decoder_kwargs, lr, scheduler_kwargs, batch_size)

    # load data
    data = datasets.OfflineData(env_name, data_dir, num_frames)
    lengths = [len(data)-int(len(data)*val_perc), int(len(data)*val_perc)]
    train_data, val_data = random_split(data, lengths)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=3)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=3)

    num_batches = len(train_data) // batch_size
    if len(train_data) % batch_size != 0:
        num_batches += 1

    print(f'\nnum train samples = {len(train_data)} --> {num_batches} train batches')
    print(f'num val samples = {len(val_data)}')

    # create callbacks to sample reconstructed images and for model checkpointing
    img_callback =  GenerateCallback(dataset=val_data, save_to_disk=False)
    checkpoint_callback = ModelCheckpoint(mode="min", monitor="Validation/bpd", save_last=True)

    trainer=pl.Trainer(
                    precision=16, #32 is normal, 16 is mixed precision
                    progress_bar_refresh_rate=100, #every N batches update progress bar
                    callbacks=[img_callback, checkpoint_callback],
                    gpus=torch.cuda.device_count(),
                    auto_lr_find=True,
                    #auto_scale_batch_size='binsearch',
                    accelerator='dp', #anything else here seems to lead to crashes/errors
                    default_root_dir=log_dir,
                    max_epochs=epochs
                )
    trainer.logger._default_hp_metric = None # optional logging metric that we don't need right now
    trainer.fit(model, train_loader, val_loader) # fit model

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--env_name')
    parser.add_argument('--model_class', choices=['Conv','CLSTM'])
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--val_perc', default=0.1, type=float, help='How much of the data should be used for validation')
    parser.add_argument('--eval_freq', default=1, type=int, help='How often to reconstruct a random val image for tensorboard')

    args = vars(parser.parse_args())

    train_VAE(**args)
