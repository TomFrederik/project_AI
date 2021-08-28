import torch
from state_vqvae import StateVQVAE
from datasets import StateVQVAEData
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os


def main(framevqvae, env_name, data_dir, batch_size, lr, epochs, save_freq, log_dir, num_workers, load_from_checkpoint, version):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # make sure that relevant dirs exist
    run_name = f'StateVQVAE/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    
    dataset = StateVQVAEData(env_name, data_dir, num_workers) # TODO: Implement
    
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    
    optim_kwargs = {
        'lr': lr
    }
    model_kwargs ={
        'optim_kwargs':optim_kwargs,
        'framevqvae':framevqvae
    }
    if load_from_checkpoint:
        checkpoint_file = os.path.join(log_dir, 'lightning_logs', f'version_{version}', 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint_file}')
        model = StateVQVAE.load_from_checkpoint(checkpoint_file, **model_kwargs)
    else:
        model = StateVQVAE(**model_kwargs).to(device)
    
    callbacks = [ModelCheckpoint(monitor='Training/loss', mode='min', every_n_train_steps=save_freq)]
    
    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        log_every_n_steps=1,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        accelerator='dp',
        default_root_dir=log_dir,
        max_epochs=epochs
    )
    
    trainer.fit(model, train_loader)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framevqvae', type=str, help='Path to the FrameVQVAE checkpoint')
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--version', type=int, default=0, help='Version of model, if training is resumed from checkpoint')
    
    args = parser.parse_args()
    
    main(**vars(args))
