from dynamics_models import MDN_RNN
import datasets

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

import numpy as np
from time import time
import os
import argparse
import einops


# for debugging
#torch.autograd.set_detect_anomaly(True)


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

        if isinstance(dataset, datasets.DynamicsData):
            iterator = iter(dataset)
            for _ in range(2000):
                b = next(iterator)
            pov, vec_obs, act = map(lambda x: x[None,:seq_len], next(iterator)[:-1])
        elif isinstance(dataset, datasets.TrajectoryData):
            pov, vec_obs, act = map(lambda x: x[None,:seq_len], dataset[3][:3])
        else:
            raise NotImplementedError

        #pov, vec_obs, act = map(lambda x: x[:,:seq_len], dataset[0][:-1])
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

    def predict_sequence(self, trainer, pl_module: MDN_RNN, epoch):
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
        if pl_module.hparams.visual_model_cls == 'vqvae':
            # one-step predictions
            logits, mixing_logits, *_ = pl_module.forward(*self.sequence)
            logits = einops.rearrange(logits, 'b t K d -> (b t) K d')
            mixing_logits = einops.rearrange(mixing_logits, 'b t K -> (b t) K')
            sampled_mix = torch.nn.functional.gumbel_softmax(mixing_logits, tau=1, hard=True, dim=-1)
            sampled_logits = torch.einsum('a b c, a b -> a c', logits, sampled_mix)
            sampled_logits = einops.rearrange(sampled_logits, 'b (n d) -> b n d', n=32, d=32)
            sampled_one_hot = torch.nn.functional.gumbel_softmax(sampled_logits, hard=True, dim=-1)

            pov_samples = []
            for i in range(sampled_one_hot.shape[1]):
                pov_samples.append(sampled_one_hot[:,i] @ pl_module.visual_model.quantizer.embeds[i].weight)
            pov_samples = torch.stack(pov_samples, dim=1)[:-1]

            # n-step predictions
            n_step_predictions = pl_module.imaginate(self.sequence[0][0][0], self.sequence[1][0][0], self.sequence[2][0])[:-1, :-64]
            n_step_predictions = einops.rearrange(n_step_predictions, 'b (n d) -> b n d', n=32, d=32)
        
        else:
            # one-step predictions
            means, log_stds, mixing_logits, *_ = pl_module.forward(*self.sequence)
            means = einops.rearrange(means, 'b t K d -> (b t) K d')
            log_stds = einops.rearrange(log_stds, 'b t K d -> (b t) K d')
            mixing_logits = einops.rearrange(mixing_logits, 'b t K -> (b t) K')
            sampled_mix = torch.argmax(torch.nn.functional.gumbel_softmax(mixing_logits, tau=1, hard=True, dim=-1),dim=1)
            sampled_means = means[torch.arange(len(means)), sampled_mix]
            sampled_log_stds = log_stds[torch.arange(len(log_stds)), sampled_mix]
            pov_samples = sampled_means + torch.exp(sampled_log_stds) * torch.normal(torch.zeros_like(sampled_means), torch.ones_like(sampled_log_stds))
            pov_samples = pov_samples[:-1]

            # n-step predictions
            n_step_predictions = pl_module.imaginate(self.sequence[0][0][0], self.sequence[1][0][0], self.sequence[2][0])[:-1, :-64]
            
        # reconstruct images
        base_reconstruction = pl_module.visual_model.reconstruct_only(self.sequence[0][0,1:])
        one_step_pov_reconstruction = pl_module.visual_model.decode_only(pov_samples)
        n_step_pov_reconstruction = pl_module.visual_model.decode_only(n_step_predictions)

        # log images to tensorboard
        images = torch.stack([self.sequence[0][0,1:], base_reconstruction, one_step_pov_reconstruction, n_step_pov_reconstruction], dim=1).reshape(((self.seq_len -1) * 4, 3, 64, 64))
        pl_module.logger.experiment.log({'Predictions (raw | base reconstruction | one-step | n-step)': wandb.Image(make_grid(images, nrow=4))})


def train_DynamicsModel(
    env_name, 
    data_dir, 
    log_dir, 
    seq_len, 
    lr, 
    batch_size, 
    use_whole_trajectories,
    num_epochs, 
    visual_model_cls, 
    visual_model_path, 
    num_components,
    gru_hidden_size,
    load_from_checkpoint, 
    checkpoint_path,
    curriculum_threshold, 
    curriculum_start,
    save_freq
):
    
    pl.seed_everything(1337)
    
    if use_whole_trajectories:
        print('\nTraining on complete trajectories, batch size is forced to 1 and seq_len will vary!\n')
        batch_size = 1
        
    # make sure that relevant dirs exist
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    ## some model kwargs
    optim_kwargs = {'lr':lr}
    gru_kwargs = {'num_layers':1, 'hidden_size':gru_hidden_size}
    model_kwargs = {
        'gru_kwargs':gru_kwargs, 
        'visual_model_path':visual_model_path,
        'optim_kwargs':optim_kwargs,
        'visual_model_cls':visual_model_cls,
        'num_components':num_components,
        'curriculum_threshold':curriculum_threshold,
        'curriculum_start':curriculum_start,
    }
    monitor = 'Training/loss'
    
    # init model
    if load_from_checkpoint:
        print(f'\nLoading model from {checkpoint_path}')
        model = MDN_RNN.load_from_checkpoint(checkpoint_path)
    else:
        model = MDN_RNN(**model_kwargs)

    # load data
    if use_whole_trajectories:
        train_data = datasets.TrajectoryData(env_name, data_dir)
    else:
        train_data = datasets.DynamicsData(env_name, data_dir, seq_len, batch_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, pin_memory=True)

    model_checkpoint = ModelCheckpoint(mode="min", monitor=monitor, save_last=True, every_n_train_steps=save_freq)
    prediction_callback = PredictionCallback(
        every_n_batches=save_freq,
        dataset=train_data,
        seq_len=10
    )
    callbacks = [model_checkpoint, prediction_callback]
    config = dict(
        env_name=env_name,
        visual_model_cls=visual_model_cls,
        dynamics_model='MDN_RNN'
    )
    wandb_logger = WandbLogger(project='Dynamics', config=config, tags=['MDN_RNN', visual_model_cls])
    trainer=pl.Trainer(
        logger=wandb_logger,
        progress_bar_refresh_rate=1, #every N batches update progress bar
        log_every_n_steps=1,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        default_root_dir=log_dir,
        max_epochs=num_epochs,
    )
    trainer.fit(model, train_loader)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', default="/home/lieberummaas/datadisk/minerl/data")
    parser.add_argument('--log_dir', default="/home/lieberummaas/datadisk/minerl/run_logs")
    parser.add_argument('--env_name', default='MineRLNavigateDenseVectorObf-v0')
    
    # training args
    parser.add_argument('--seq_len', default=10, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--use_whole_trajectories', action='store_true', help='Train on complete trajectories instead of subsequences -> batch size is forced to 1!')
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--save_freq', default=100, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    
    # sequence learning args
    # parser.add_argument('--latent_overshooting', action='store_true')
    parser.add_argument('--curriculum_threshold', default=3, type=float)
    parser.add_argument('--curriculum_start', default=0, type=int)

    # visual model args
    parser.add_argument('--visual_model_cls', type=str, default='vae', choices=['vae', 'vqvae'])
    parser.add_argument('--visual_model_path', type=str, required=True)
    
    # MDN-RNN args
    parser.add_argument('--num_components', type=int, default=5, help='Number of mixture components. Only used in MDN-RNN')
    parser.add_argument('--gru_hidden_size', type=int, default=512, help='Hidden size of the gru')

    args = vars(parser.parse_args())

    train_DynamicsModel(**args)
