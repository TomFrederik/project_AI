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
        #return self.decoder(z)

    @torch.no_grad()
    def encode_only(self, x):
        '''
        Encodes images into their latent space (via sampling).
        Does not track gradients. Use e.g. as preprocessing/embedding.
        Args:
            x - input, for mineRL (B, C, H, W) batch of frames
        Returns:
            mean
            log_std
            sample - tensor of shape (B, L), where L is the latent dimension
        '''
        b, *_ = x.shape
        mean, log_std = self.encoder(x-0.5)
        

        sample = self.sample(mean, log_std)
        return sample, mean, log_std
    
    def encode_with_grad(self, x):
        b, *_ = x.shape
        mean, log_std = self.encoder(x-0.5)
        
        # compute KL distance, i.e. regularization loss
        L_regul = (0.5 * (torch.exp(2 * log_std) + mean ** 2 - 1 - 2 * log_std)).sum(dim=-1).mean()

        sample = self.sample(mean, log_std)
        return sample, L_regul, None, None

    @torch.no_grad()
    def decode_only(self, z):
        #return self.decoder(z)
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
        #x_hat = self.decoder(z)
        
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
        recon_loss, latent_loss = self(obs)

        loss = recon_loss + self.beta * latent_loss

        self.log('Training/loss', loss, on_step=True)
        self.log('Training/recon_loss', recon_loss, on_step=True)
        self.log('Training/latent_loss', latent_loss, on_step=True)

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
            nn.Conv2d(3, 32, 4, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(1024, 2*latent_dim)
        )
        '''
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4), # --> shape is (...., 16, 16)
            nn.ReLU(),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, stride=2, padding=1), # (.... 4, 4)
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(2*n_hid*16, 2*latent_dim)
        )'''

    def forward(self, x):
        #out = self.net(x)
        #print('\nEncoder:')
        out = x
        for m in self.net:
        #    print(out.shape)
            out = m(out)
        #print(out.shape)
        mean, log_std = torch.chunk(out, chunks=2, dim=-1)        
        return mean, log_std


class VAEDecoder(nn.Module):

    def __init__(self, latent_dim=64, n_init=64, n_hid=64, output_channels=3):
        super().__init__()

        '''
        self.net = nn.Sequential(
            Rearrange('b (h w c) -> b c h w', w=4, h=4),
            nn.Conv2d(64, n_init, 3, padding=1),
            nn.UpsamplingNearest2d((8,8)),
            nn.ReLU(),
            nn.Conv2d(n_init, n_init, 3, padding=1),
            nn.UpsamplingNearest2d((16,16)),
            nn.ReLU(),
            nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )
        '''
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            Rearrange('b d -> b d 1 1'),
            nn.ConvTranspose2d(1024, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2),
            #nn.Sigmoid(),
        )
        

    def forward(self, x):
        #print('\nDecoder:')
        out = x
        for m in self.net:
        #    print(out.shape)
            out = m(out)
        #print(out.shape)
        return out
        #return self.net(x)
