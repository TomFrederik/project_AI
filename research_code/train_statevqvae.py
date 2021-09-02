import torch
from state_vqvae import StateVQVAE
from datasets import StateVQVAEData
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
from time import time

"""
These ramps/decays follow DALL-E Appendix A.2 Training https://arxiv.org/abs/2102.12092
"""
class DecayTemperature(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
        t = cos_anneal(0, 150000, 1.0, 1.0/16, trainer.global_step)
        pl_module.quantizer.temperature = t

class RampBeta(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        t = cos_anneal(0, 5000, 0.0, 5e-4, trainer.global_step)
        pl_module.quantizer.kld_scale = t

class DecayLR(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, 1200000, 3e-4, 1.25e-6, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t
            
            
def main(framevqvae, env_name, data_dir, batch_size, lr, epochs, save_freq, log_dir, num_workers, load_from_checkpoint, version, num_trajs, embedding_dim, codebook_size, gumbel, tau):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # make sure that relevant dirs exist
    run_name = f'StateVQVAE/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    dataset = StateVQVAEData(env_name, data_dir, num_workers, num_trajs) # TODO: Implement
    
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    optim_kwargs = {
        'lr': lr
    }
    model_kwargs ={
        'optim_kwargs':optim_kwargs,
        'framevqvae':framevqvae,
        'embedding_dim':embedding_dim,
        'codebook_size':codebook_size,
        'gumbel':gumbel,
        'tau':tau
    }
    if load_from_checkpoint:
        checkpoint_file = os.path.join(log_dir, 'lightning_logs', f'version_{version}', 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint_file}')
        model = StateVQVAE.load_from_checkpoint(checkpoint_file, **model_kwargs)
    else:
        model = StateVQVAE(**model_kwargs).to(device)
    
    '''
    stat_path = os.path.join(framevqvae[:-9], 'stats.json')
    print(f'{stat_path = }')
    if os.path.exists(stat_path):
        model.find_data_mean_var(train_loader, load_from=stat_path)
    else:
        model.find_data_mean_var(train_loader, save_to=stat_path)
    '''
    
    callbacks = [ModelCheckpoint(monitor='Training/reconstruction_loss', mode='min', every_n_train_steps=save_freq, save_last=True)]
    # TODO add callbacks for LR decay and beta ramp
    callbacks.append(DecayLR())
    if args.vq_flavor == 'gumbel':
       callbacks.extend([DecayTemperature(), RampBeta()])
    
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
    parser.add_argument('--num_trajs', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--version', type=int, default=0, help='Version of model, if training is resumed from checkpoint')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--codebook_size', type=int, default=32)
    parser.add_argument('--gumbel', action='store_true')
    parser.add_argument('--tau', type=float, default=1)
    
    args = parser.parse_args()
    
    main(**vars(args))