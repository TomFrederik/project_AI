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

# for debugging
torch.autograd.set_detect_anomaly(True)

STR_TO_MODEL = {
    'rssm':dynamics_models.RSSM,
    'node':dynamics_models.NODEDynamicsModel,
    'mdn':dynamics_models.MDN_RNN
}

def train_DynamicsModel(env_name, data_dir, dynamics_model, seq_len, lr, 
                        val_perc, batch_size, num_data, epochs, 
                        lr_gamma, lr_decrease_freq, log_dir, lr_step_mode, 
                        model_path, VAE_class, num_components, skip_connection,
                        val_check_interval, load_from_checkpoint, version_dir,
                        latent_overshooting, soft_targets, profile, temp, regression,
                        conditioning_len, curriculum_threshold, curriculum_start):
    
    # make sure that relevant dirs exist
    run_name = f'DynamicsModel/{STR_TO_MODEL[dynamics_model].__name__}/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    ## some model kwargs
    optim_kwargs = {'lr':lr}
    scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
    
    if dynamics_model == 'node':
        seq_len = seq_len
        hidden_dims = [512,512,512]
        base_model_class = dynamics_models.DynamicsBaseModel
        base_model_kwargs = {'input_dim':256, 'hidden_dims':hidden_dims}
        
        model_kwargs = {
            'base_model_class':base_model_class, 
            'base_model_kwargs':base_model_kwargs, 
            'seq_len':seq_len, 
            'VAE_path':model_path,
            'optim_kwargs':optim_kwargs,
            'scheduler_kwargs':scheduler_kwargs
        }
        monitor = 'Validation/loss'
    elif dynamics_model == 'rssm':
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
    elif dynamics_model == 'mdn':
        seq_len = seq_len        
        gru_kwargs = {'num_layers':1, 'hidden_size':1024}
        model_kwargs = {
            'gru_kwargs':gru_kwargs, 
            'seq_len':seq_len, 
            'VAE_path':model_path,
            'optim_kwargs':optim_kwargs,
            'scheduler_kwargs':scheduler_kwargs,
            'VAE_class':VAE_class,
            'num_components':num_components,
            'temp':temp,
            'skip_connection':skip_connection,
            'latent_overshooting':latent_overshooting,
            'soft_targets':soft_targets,
            'conditioning_len':conditioning_len,
            'curriculum_threshold':curriculum_threshold,
            'curriculum_start':curriculum_start
        }
        monitor = 'Validation/loss'
    else:
        ValueError(f"Unrecognized model {dynamics_model}")
    ##
    
    # init model
    if load_from_checkpoint:
        checkpoint = os.path.join(version_dir, 'checkpoints', 'last.ckpt')
        
        print(f'\nLoading model from {checkpoint}')
        model = STR_TO_MODEL[dynamics_model].load_from_checkpoint(checkpoint, lr=lr, curriculum_start=curriculum_start)
    else:
        model = STR_TO_MODEL[dynamics_model](**model_kwargs)

    # load data
    data = datasets.DynamicsData(env_name, data_dir, seq_len + conditioning_len, num_data)
    lengths = [len(data)-int(len(data)*val_perc), int(len(data)*val_perc)]
    train_data, val_data = random_split(data, lengths)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory=True)

    num_batches = len(train_data) // batch_size
    if len(train_data) % batch_size != 0:
        num_batches += 1

    print(f'\nnum train samples = {len(train_data)} --> {num_batches} train batches')
    print(f'num val samples = {len(val_data)}')

    model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="min", monitor=monitor, save_last=True)
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
    trainer.logger._default_hp_metric = None # optional logging metric that we don't need right now
    trainer.fit(model, train_loader, val_loader)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', help='Path to encoding model')
    parser.add_argument('--data_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--env_name')
    parser.add_argument('--dynamics_model', default='rssm', choices=['rssm', 'node', 'mdn'], help='Model used to predict the next latent state')
    parser.add_argument('--seq_len', default=4, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_data', default=0, type=int, help='Number of datapoints to use')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_gamma', default=1, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--val_perc', default=0.1, type=float, help='How much of the data should be used for validation')
    parser.add_argument('--VAE_class', type=str, default='Conv', choices=['Conv', 'ResNet', 'vqvae'])
    parser.add_argument('--num_components', type=int, default=5, help='Number of mixture components. Only used in MDN-RNN')
    parser.add_argument('--skip_connection', action='store_true', help='Whether to use skip connection in MDN-RNN.')
    parser.add_argument('--val_check_interval', default=1, type=int, help='How often to validate. N == 1 --> once per epoch; N > 1 --> every N steps')
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--latent_overshooting', action='store_true')
    parser.add_argument('--soft_targets', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--regression', action='store_true', help='Whether to perform regression or KL minimization')
    parser.add_argument('--temp', default=1, type=float)
    parser.add_argument('--curriculum_threshold', default=3, type=float)
    parser.add_argument('--curriculum_start', default=0, type=int)
    parser.add_argument('--conditioning_len', default=0, type=int, help='Length of sequence to condition rnn on')
    parser.add_argument('--version_dir', default='', type=str, help='Version directory of model, if training is resumed from checkpoint')

    args = vars(parser.parse_args())

    train_DynamicsModel(**args)
