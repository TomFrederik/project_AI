import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.optim import AdamW
import torch.nn.functional as F

import einops
from einops.layers.torch import Rearrange

class VAE(pl.LightningModule):
    '''
    A base class for VAEs
    '''
    def __init__(self, encoder_kwargs, decoder_kwargs, learning_rate, beta=1):
        '''
        beta - positive float which controls the strength of the regularization, as described in beta-VAE paper. 
        '''
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta

        self.encoder = VAEEncoder(**encoder_kwargs)
        self.decoder = VAEDecoder(**decoder_kwargs)

    def sample(self, mean, log_std):
        '''
        Implements reparameterizatio trick to sample from the given normal distribution
        '''
        z = mean + torch.exp(log_std) * torch.normal(torch.zeros_like(mean), torch.ones_like(log_std))
        return z
    
    @torch.no_grad()
    def reconstruct_only(self, x):
        '''
        Encodes x, samples z and reconstructs x
        x - shape (B, C, H, W)
        '''
        # encode
        mean, log_std = self.encoder(x - 0.5)

        # sample latent vector
        z = self.sample(mean, log_std)
        return torch.clamp(0.5 + self.decoder(z), 0, 1)

    @torch.no_grad()
    def encode_only(self, x):
        '''
        Encodes images into their latent space (via sampling).
        Does not track gradients. Use e.g. as preprocessing/embedding.
        Args:
            x - input, for mineRL (B, C, H, W) batch of frames
        Returns:
            mean
            std
            sample - tensor of shape (B, L), where L is the latent dimension
        '''
        b, *_ = x.shape
        mean, log_std = self.encoder(x-0.5)
        h = int((mean.shape[0]//b) ** 0.5)
        
        mean = einops.rearrange(mean, '(b h w) c -> b c h w', b=b, h=h, w=h)
        log_std = einops.rearrange(log_std, '(b h w) c -> b c h w', b=b, h=h, w=h)

        sample = self.sample(mean, log_std)
        return mean, torch.exp(log_std), sample

    @torch.no_grad()
    def decode_only(self, z):
        return torch.clamp(0.5 + self.decoder(z), 0, 1)

    def forward(self, x):
        '''
        x - input, for mineRL (B, C, H, W) batch of frames
        '''
        # encode
        mean, log_std = self.encoder(x - 0.5)

        # compute KL distance, i.e. regularization loss
        L_regul = (0.5 * (torch.exp(2 * log_std) + mean ** 2 - 1 - 2 * log_std)).sum(dim=-1).mean()

        # sample latent vector
        z = self.sample(mean, log_std)
        
        # decode
        x_hat = torch.clamp(self.decoder(z) + 0.5, 0, 1)
        
        # compute reconstruction loss, sum over all dimension except batch
        L_reconstr = (x - x_hat).pow(2).mean() / (2* 0.06327039811675479) # cifar-10 data variance, from deepmind sonnet code)

        return L_reconstr, L_regul
    
    def configure_optimizers(self):
        # set up optimizer
        self.optimizer =  AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return self.optimizer
    
    def training_step(self, batch, batch_idx):
        obs, *_ = batch
        obs = obs['pov'].float() / 255
        obs = einops.rearrange(obs, 'b h w c -> b c h w')
        L_rec, L_reg = self(obs)

        loss = L_rec + self.beta * L_reg

        self.log('Training/loss', loss, on_step=True)
        self.log('Training/recon_loss', L_rec, on_step=True)
        self.log('Training/latent_loss', L_reg, on_step=True)

        return loss


class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out


class VAEEncoder(nn.Module):

    def __init__(self, input_channels=3, n_hid=64, latent_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            Rearrange('b c h w -> (b h w) c'),
            nn.Linear(2*n_hid, 2*latent_dim)
        )

    def forward(self, x):
        out = self.net(x)
        mean, log_std = torch.chunk(out, chunks=2, dim=-1)        
        return mean, log_std


class VAEDecoder(nn.Module):

    def __init__(self, latent_dim=64, n_init=64, n_hid=64, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, n_init),
            Rearrange('(b h w) c -> b c h w', w=16, h=16),
            nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)
