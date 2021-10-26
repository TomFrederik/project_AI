import argparse
from copy import deepcopy
import os
import random 

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
    num_expert_episodes,
    visual_model_cls, 
    visual_model_path, 
    unfreeze_visual_model,
    dynamics_model_cls, 
    dynamics_model_path, 
    unfreeze_dynamics_model,
    centroids_path, 
    num_centroids,
    target_update_rate, 
    margin, 
    discount_factor, 
    horizon,
    use_one_hot
):
    # random seed
    pl.seed_everything(1337)
    random.seed(1337)

    # some sanity checks
    if batch_size > 1: 
        raise NotImplementedError
    
    if visual_model_cls == 'conv' and not unfreeze_visual_model:
        raise ValueError("Mustn't freeze_visual_model when using conv!")

    # make sure that relevant dirs exist
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')
    
    # set up WandB logger
    config = dict(
        env_name=env_name,
        visual_model_cls=visual_model_cls,
        visual_model_path=visual_model_path,
        freeze_visual_model=not unfreeze_visual_model,
        dynamics_model_cls=dynamics_model_cls,
        dynamics_model_path=dynamics_model_path,
        freeze_dynamics_model=not unfreeze_dynamics_model,
        use_one_hot=use_one_hot
    )
    if dynamics_model_cls is not None:
        wandb_logger = WandbLogger(project='DQfD_pretraining', config=config, tags=[visual_model_cls, dynamics_model_cls, 'one_hot_'+str(use_one_hot)])
    else:
        wandb_logger = WandbLogger(project='DQfD_pretraining', config=config, tags=[visual_model_cls, 'one_hot_'+str(use_one_hot)])

    # load centroids for action discretization
    centroids_path = os.path.join(centroids_path, env_name + f'_{num_centroids}_centroids.npy')
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
        'freeze_visual_model':~unfreeze_visual_model,
        'dynamics_model_cls':dynamics_model_cls, 
        'dynamics_model_path':dynamics_model_path, 
        'freeze_dynamics_model':not unfreeze_dynamics_model,
        'use_one_hot':use_one_hot
    }
    
    # set up model
    model = QNetwork(**model_kwargs)
    
    # load data
    data = datasets.TrajectoryData(env_name, data_dir, num_expert_episodes, centroids)
    train_data, val_data = torch.utils.data.random_split(data, [int(len(data)*0.9), len(data) - int(len(data)*0.9)])
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # set up trainer
    model_checkpoint = ModelCheckpoint(mode="min", monitor='Training/Loss', save_last=True, every_n_train_steps=500)
    trainer=pl.Trainer(
        logger=wandb_logger,
        progress_bar_refresh_rate=1, #every N batches update progress bar
        log_every_n_steps=100,
        callbacks=[model_checkpoint],
        gpus=torch.cuda.device_count(),
        default_root_dir=log_dir,
        max_epochs=epochs,
        #track_grad_norm=2
    )

    # train
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLNavigateDenseVectorObf-v0')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--centroids_path', type=str, default='/home/lieberummaas/datadisk/minerl/data/')
    parser.add_argument('--num_centroids', type=int, default=1000)
    
    # training args
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--num_expert_episodes', default=300, type=int)
    parser.add_argument('--target_update_rate', default=100, type=int, help='How often to update target network')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    
    # Q-learning args
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--margin', default=0.8, type=float)
    parser.add_argument('--horizon', default=50, type=int, help='Horizon for n-step TD error')
    parser.add_argument('--use_one_hot', action='store_true', help='whether to use one-hot representation')
    
    # feature extractor args
    parser.add_argument('--visual_model_cls', choices=['vqvae', 'vae', 'conv'], default='vae', help='Class of the visual_model model')
    parser.add_argument('--visual_model_path', help='Path to visual_model model')
    parser.add_argument('--unfreeze_visual_model', action='store_true', help='Whether to freeze/finetune the visual mode')
    
    # dynamics model args
    parser.add_argument('--dynamics_model_cls', choices=['mdn', None], default=None, help='Class of the dynamics model')
    parser.add_argument('--dynamics_model_path', default=None, help='Path to dynamics model')
    parser.add_argument('--unfreeze_dynamics_model', action='store_true', help='Whether to freeze/finetune the dynamics model')
    
    
    args = parser.parse_args()
    
    main(**vars(args))