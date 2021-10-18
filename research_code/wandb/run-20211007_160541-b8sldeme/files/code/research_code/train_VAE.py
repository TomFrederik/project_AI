from time import time
import os
import argparse
import math

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import einops

from vae_model import VAE
import datasets

# -----------------------------------------------------------------------------
def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

class RampBeta(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        t = cos_anneal(0, 15000, 0.0, 5e-4, trainer.global_step)
        pl_module.beta = t

class DecayLR(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, 120000000, 3e-4, 1.25e-6, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t

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
        self.img_batch = next(self.dataset.iter.buffered_batch_iter(batch_size, num_batches=1))[0]['pov']
        self.img_batch = torch.from_numpy(einops.rearrange(self.img_batch, 'b h w c -> b c h w')).float() / 255

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
        
        if self.img_batch.device != pl_module.device:
            self.img_batch = self.img_batch.to(pl_module.device)

        reconstructed_img = pl_module.reconstruct_only(self.img_batch)

        images = torch.stack([self.img_batch, reconstructed_img], dim=1).reshape((self.batch_size * 2, *self.img_batch.shape[1:]))

        # log images to tensorboard
        pl_module.logger.experiment.log({'Reconstruction': wandb.Image(make_grid(images, nrow=2))})


def train_VAE(
    env_name, 
    data_dir, 
    lr,
    eval_freq, 
    save_freq, 
    batch_size,
    epochs, 
    log_dir, 
    latent_dim, 
    n_hid,
    n_init
):
    pl.seed_everything(1337)

    # make sure that relevant dirs exist
    run_name = f'VAE/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'Saving logs and model to {log_dir}')

    encoder_kwargs = {
        'latent_dim':latent_dim,
        'n_hid':n_hid,
    }
    decoder_kwargs = {
        'latent_dim':latent_dim,
        'n_hid':n_hid,
        'n_init':n_init
    }
    
    # init model
    model = VAE(encoder_kwargs, decoder_kwargs, lr)
    
    # load data
    train_data = datasets.BufferedBatchDataset(env_name, data_dir, batch_size, epochs)
    train_loader = DataLoader(train_data, batch_size=None, num_workers=1)

    # create callbacks to sample reconstructed images and for model checkpointing
    img_callback =  GenerateCallback(dataset=train_data, save_to_disk=False)
    checkpoint_callback = ModelCheckpoint(mode="min", monitor="Training/loss", save_last=True, every_n_train_steps=save_freq)
    callbacks = [img_callback, checkpoint_callback, RampBeta()]
    # callbacks.append(DecayLR())
    # init logger
    wandb_logger = WandbLogger(project="VAE", log_model="all")
    trainer=pl.Trainer(
        logger=wandb_logger,
        progress_bar_refresh_rate=10, #every N batches update progress bar
        log_every_n_steps=10,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        #accelerator='ddp', #anything else here seems to lead to crashes/errors
        default_root_dir=log_dir,
        max_epochs=epochs,
    )
                    
    # fit model
    trainer.fit(model, train_loader)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--env_name', default='MineRLNavigateDenseVectorObf-v0')
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--latent_dim', default=1024, type=int)
    parser.add_argument('--n_hid', default=64, type=int)
    parser.add_argument('--n_init', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--eval_freq', default=1, type=int, help='How often to reconstruct images for tensorboard')
    parser.add_argument('--save_freq', default=100, type=int, help='How often to save model')

    args = vars(parser.parse_args())

    train_VAE(**args)
