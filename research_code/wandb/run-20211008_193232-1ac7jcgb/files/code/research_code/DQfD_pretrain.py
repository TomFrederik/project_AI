import argparse
from copy import deepcopy
import os

import einops
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import wandb

import datasets
from DQfD_models import QNetwork, ConvFeatureExtractor
from dynamics_models import MDN_RNN
from vae_model import VAE
from vqvae import VQVAE

def main(
    env_name, 
    batch_size, 
    num_workers, 
    lr, 
    weight_decay, 
    data_dir, 
    log_dir,
    epochs, 
    visual_model_cls, 
    visual_model_path, 
    freeze_visual_model,
    dynamics_model_cls, 
    dynamics_model_path, 
    freeze_dynamics_model,
    centroids_path, 
    target_update_rate, 
    margin, 
    discount_factor, 
    horizon
):
    # some sanity checks
    if batch_size > 1: 
        raise NotImplementedError
    
    if visual_model_cls == 'conv' and freeze_visual_model:
        raise ValueError("Mustn't freeze_visual_model when using conv!")
  
    # random seed
    pl.seed_everything(1337)

    # make sure that relevant dirs exist
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')
    
    # set up WandB logger
    config = dict(
        env_name=env_name,
        visual_model_cls=visual_model_cls,
        visual_model_path=visual_model_path,
        freeze_visual_model=freeze_visual_model,
        dynamics_model_cls=dynamics_model_cls,
        dynamics_model_path=dynamics_model_path,
        freeze_dynamics_model=freeze_dynamics_model
    )
    if dynamics_model_cls is not None:
        wandb_logger = WandbLogger(project='DQfD_pretraining', config=config, tags=[visual_model_cls, dynamics_model_cls])
    else:
        wandb_logger = WandbLogger(project='DQfD_pretraining', config=config, tags=[visual_model_cls])

    # load centroids for action discretization
    centroids_path = os.path.join(centroids_path, env_name + '_150_centroids.npy')
    centroids = np.load(centroids_path)
    print(f'\nLoaded centroids from {centroids_path}! Shape is {centroids.shape}.')

    ## some model kwargs
    optim_kwargs = {'lr':lr, 'weight_decay':weight_decay}
    
    visual_model_cls = {
        'vqvae':VQVAE,
        'vae':VAE,
        'conv':ConvFeatureExtractor
    }[visual_model_cls]
    
    dynamics_model_cls = {
        'mdn':MDN_RNN,
        None:None
    }[dynamics_model_cls]
    
    model_kwargs = {
        'visual_model_path':visual_model_path,
        'optim_kwargs':optim_kwargs,
        'n_actions':centroids.shape[0],
        'target_update_rate':target_update_rate,
        'margin':margin,
        'discount_factor':discount_factor,
        'horizon':horizon,
        'visual_model_cls':visual_model_cls,
        'freeze_visual_model':freeze_visual_model,
        'dynamics_model_cls':dynamics_model_cls, 
        'dynamics_model_path':dynamics_model_path, 
        'freeze_dynamics_model':freeze_dynamics_model,
    }
    
    # set up model
    model = QNetwork(**model_kwargs)
    
    # load data
    train_data = datasets.TrajectoryData(env_name, data_dir, centroids)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # set up trainer
    model_checkpoint = ModelCheckpoint(mode="min", monitor='Training/Loss', save_last=True, every_n_train_steps=500)
    trainer=pl.Trainer(
        logger=wandb_logger,
        progress_bar_refresh_rate=1, #every N batches update progress bar
        log_every_n_steps=10,
        callbacks=[model_checkpoint],
        gpus=torch.cuda.device_count(),
        default_root_dir=log_dir,
        max_epochs=epochs
    )

    # train
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLNavigateDenseVectorObf-v0')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--centroids_path', type=str, default='/home/lieberummaas/datadisk/minerl/data/')
    
    # training args
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--target_update_rate', default=100, type=int, help='How often to update target network')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    
    # Q-learning args
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--margin', default=0.8, type=float)
    parser.add_argument('--horizon', default=10, type=int, help='Horizon for n-step TD error')
    
    # feature extractor args
    parser.add_argument('--visual_model_cls', choices=['vqvae', 'vae', 'conv'], default='vae', help='Class of the visual_model model')
    parser.add_argument('--visual_model_path', help='Path to visual_model model')
    parser.add_argument('--freeze_visual_model', action='store_true', help='Whether to freeze or finetune the feature extractor')
    
    # dynamics model args
    parser.add_argument('--dynamics_model_cls', choices=['mdn', None], default=None, help='Class of the dynamics model')
    parser.add_argument('--dynamics_model_path', default=None, help='Path to dynamics model')
    parser.add_argument('--freeze_dynamics_model', action='store_true', help='Whether to freeze or finetune the dynamics model extractor')
    
    
    args = parser.parse_args()
    
    main(**vars(args))