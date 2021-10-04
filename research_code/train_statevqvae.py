import torch
from state_vqvae import StateVQVAE
from datasets import StateVQVAEData
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
from time import time
import math
import einops
from torchvision.utils import make_grid


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
        pov, vec_obs, act = map(lambda x: x[None,:seq_len], dataset[0])
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

    def predict_sequence(self, trainer, pl_module:StateVQVAE, epoch):
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
        #extrapolate_input = {
        #    'pov_obs':self.sequence[0][:,0][:,None],
        #    'vec_obs':self.sequence[1][:,0][:,None],
        #    'actions':self.sequence[2]
        #}
        #predictions, *_ = pl_module.extrapolate(**extrapolate_input)
        predictions, *_ = pl_module(*self.sequence)
        
        predictions = torch.nn.functional.softmax(predictions[0], dim=2)

        print(f'{predictions.shape = }')
        B, T, C, H, W = predictions.shape
        # sample from predictions
        ind_samples = torch.multinomial(einops.rearrange(predictions, 'b t c h w -> (b t h w) c'), 1)[:,0]
        print(f'{ind_samples.shape = }')
        pov_samples = torch.nn.functional.embedding(ind_samples, pl_module.vqvae.quantizer.embed.weight)
        print(f'{pov_samples.shape = }')
        pov_samples = einops.rearrange(pov_samples, '(b t h w) c -> (b t) c h w', b=B, t=T, h=H, w=W)
        print(f'{pov_samples.shape = }')
        # reconstruct images
        pov_reconstruction = pl_module.vqvae.decode_only(pov_samples)
        
        # stack images
        images = torch.stack([self.sequence[0][0,1:], pov_reconstruction], dim=1).reshape(((self.seq_len -1) * 2, 3, 64, 64))

        # log images to tensorboard
        trainer.logger.experiment.add_image('Prediction', make_grid(images, nrow=2), epoch)



"""
These ramps/decays follow DALL-E Appendix A.2 Training https://arxiv.org/abs/2102.12092
"""
def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

class DecayTemperature(pl.Callback):
    def __init__(self, max_time = 150000):
        super().__init__()
        self.max_time = max_time
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
        t = cos_anneal(0, self.max_time, 1.0, 1.0/16, trainer.global_step)
        pl_module.quantizer.temperature = t

class RampBeta(pl.Callback):
    def __init__(self, max_time = 5000):
        super().__init__()
        self.max_time = max_time
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        t = cos_anneal(0, self.max_time, 0.0, 5e-4, trainer.global_step)
        pl_module.quantizer.kld_scale = t

class DecayLR(pl.Callback):
    def __init__(self, max_time = 1200000):
        super().__init__()
        self.max_time = max_time
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, self.max_time, 3e-4, 1.25e-6, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t
            
            
def main(
    framevqvae, 
    env_name, 
    data_dir, 
    batch_size, 
    lr, 
    epochs, 
    save_freq, 
    img_callback_freq,
    log_dir, 
    num_workers, 
    load_from_checkpoint, 
    version, 
    num_trajs, 
    embedding_dim, 
    codebook_size, 
    gumbel, 
    tau,
    action_quantizer,
    vecobs_quantizer,
    lr_decay_max_time,
    ramp_beta_max_time,
    temp_decay_max_time,
    discard_priors,
    lstm_enc_input_size,
    lstm_enc_hidden_size,
    lstm_dec_hidden_size,
    max_seq_len
):
    pl.seed_everything(1337)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # get full paths for quantizers
    if action_quantizer is not None:
        action_quantizer = os.path.join(log_dir, 'ActionVQVAE', f'{env_name}', 'lightning_logs', f'version_{action_quantizer}', 'checkpoints', 'last.ckpt')
    if vecobs_quantizer is not None:
        vecobs_quantizer = os.path.join(log_dir, 'VecObsVQVAE', f'{env_name}', 'lightning_logs', f'version_{vecobs_quantizer}', 'checkpoints', 'last.ckpt')
    
    # make sure that relevant dirs exist
    run_name = f'StateVQVAE/{env_name}'
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    dataset = StateVQVAEData(env_name, data_dir, num_workers, num_trajs) 
    
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
        'tau':tau,
        'action_quantizer':action_quantizer,
        'vecobs_quantizer':vecobs_quantizer,
        'discard_priors':discard_priors,
        "lstm_enc_input_size":lstm_enc_input_size,
        "lstm_enc_hidden_size":lstm_enc_hidden_size,
        "lstm_dec_hidden_size":lstm_dec_hidden_size,
        "max_seq_len":max_seq_len
    }
    if load_from_checkpoint:
        checkpoint_file = os.path.join(log_dir, 'lightning_logs', f'version_{version}', 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint_file}')
        model = StateVQVAE.load_from_checkpoint(checkpoint_file, **model_kwargs)
    else:
        model = StateVQVAE(**model_kwargs).to(device)
    
    callbacks = [ModelCheckpoint(monitor='Training/reconstruction_loss', mode='min', every_n_train_steps=save_freq, save_last=True)]
    callbacks.append(DecayLR(lr_decay_max_time))
    if gumbel:
       callbacks.extend([DecayTemperature(temp_decay_max_time), RampBeta(ramp_beta_max_time)])
    
    callbacks.append(PredictionCallback(dataset=dataset, every_n_batches=img_callback_freq))

    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        log_every_n_steps=1,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        default_root_dir=log_dir,
        max_epochs=epochs
    )
    
    trainer.fit(model, train_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framevqvae', type=str, help='Path to the FrameVQVAE checkpoint')
    parser.add_argument('--action_quantizer', type=int, default=None, help='Version of the ActionVQVAE to use')
    parser.add_argument('--vecobs_quantizer', type=int, default=None, help='Version of the VecObsVQVAE to use')
    parser.add_argument('--env_name', type=str, default='MineRLNavigateDenseVectorObf-v0')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_trajs', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--img_callback_freq', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--version', type=int, default=0, help='Version of model, if training is resumed from checkpoint')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--codebook_size', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=10)
    parser.add_argument('--lstm_enc_input_size', type=int, default=512)
    parser.add_argument('--lstm_enc_hidden_size', type=int, default=512)
    parser.add_argument('--lstm_dec_hidden_size', type=int, default=512)
    parser.add_argument('--gumbel', action='store_true')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--lr_decay_max_time', type=int, default=1200)
    parser.add_argument('--ramp_beta_max_time', type=int, default=500)
    parser.add_argument('--temp_decay_max_time', type=int, default=1500)
    parser.add_argument('--discard_priors', action='store_true')
    
    args = parser.parse_args()

    assert args.gumbel, "Not using gumbel softmax is not implemented!"
    
    main(**vars(args))