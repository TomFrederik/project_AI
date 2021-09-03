"""
Defines the full (PyTorch Lightning module) VQVAE, which incorporates an
encoder, decoder and a quantize layer in the middle for the discrete bottleneck.
"""

import os
import math
from argparse import ArgumentParser, Namespace

import numpy as np

from torchvision.utils import make_grid

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vqvae_model.loss import Normal, LogitLaplace

import datasets

import einops 

torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
class ActionGumbelQuantizer(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, codebook_size, embedding_dim, latent_size, straight_through=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.latent_size = latent_size

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 10

        self.embedding = nn.Embedding(codebook_size, embedding_dim)

    def forward(self, z):
        z = einops.rearrange(z, 'b (codebook latent_size) -> b latent_size codebook', codebook=self.codebook_size)

        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True
        soft_one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=2, hard=hard)
        #print(f'{z.shape = }')
        #print(f'{soft_one_hot.shape = }')
        #print(f'{self.embedding.weight.shape = }')
        z_q = soft_one_hot @ self.embedding.weight
        #z_q = einsum('b lat codebook, codebook embed -> b lat embed', soft_one_hot, self.embedding.weight)
        #print(f'{z_q.shape = }')

        # + kl divergence to the prior loss
        qy = F.softmax(z, dim=2)
        latent_loss = self.kld_scale * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=2).mean()
        ind = soft_one_hot.argmax(dim=2)
        z_q = einops.rearrange(z_q, 'b l e -> b (l e)')
        ind = einops.rearrange(ind, 'b l -> (b l)')

        return z_q, latent_loss, ind



class ActionVQVAE(pl.LightningModule):

    def __init__(self, input_dim, codebook_size, embedding_dim, latent_size, log_perplexity, perplexity_freq, num_hidden=512):
        super().__init__()
        self.save_hyperparameters()

        # encoder/decoder module pair
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, codebook_size*latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim*latent_size, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, input_dim),
            nn.Tanh() # action space is roughly boxed in [-1, 1]
        ) 

        # the quantizer module sandwiched between them, +contributes a KL(posterior || prior) loss to ELBO
        self.quantizer = ActionGumbelQuantizer(codebook_size, embedding_dim, latent_size)

        # the data reconstruction loss in the ELBO
        self.recon_loss = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        z_q, latent_loss, ind = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, ind

    
    @torch.no_grad()
    def reconstruct_only(self, x):
        z = self.encoder(x)
        z_q, _, _ = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat
    
    @torch.no_grad()
    def decode_only(self, z_q):
        x_hat = self.decoder(z_q)
        return x_hat

    @torch.no_grad()
    def encode_only(self, x):
        z = self.encoder(x)
        z_q, _, ind = self.quantizer(z)
        return z_q, ind
    
    def encode_with_grad(self, x):
        z = self.encoder(x)
        z_q, diff, ind = self.quantizer(z)
        return z_q, diff, ind
    
    def training_step(self, batch, batch_idx):
        _, action, *_ = batch
        action = action['vector'].float()[0]
        prediction, latent_loss, ind = self.forward(action)
        recon_loss = self.recon_loss(prediction, action)
        loss = recon_loss + latent_loss
        self.log('Training/loss', loss, on_step=True)
        self.log('Training/reconstruction_loss', recon_loss, on_step=True)
        self.log('Training/latent_loss', latent_loss, on_step=True)
        if self.hparams.log_perplexity:
            if (self.global_step + 1) % self.hparams.perplexity_freq == 0:
                self.eval()
                perplexity, cluster_use = self._compute_perplexity(ind)
                self.train()
                self.log('Training/perplexity', perplexity, prog_bar=True)
                self.log('Training/cluster_use', cluster_use, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        self.optimizer = optimizer
        return optimizer
    
    @torch.no_grad()
    def _compute_perplexity(self, ind):
        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(ind, self.quantizer.codebook_size).float().reshape(-1, self.quantizer.codebook_size)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        return perplexity, cluster_use
# -----------------------------------------------------------------------------
def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

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




def main():
    pl.seed_everything(1337)

    # -------------------------------------------------------------------------
    # arguments...
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--log_freq', type=int, default=10, help='How often to save values to the logger')
    parser.add_argument('--log_perplexity', action='store_true')
    parser.add_argument('--perplexity_freq', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_trajs', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--version', type=int, default=0, help='Version of model, if training is resumed from checkpoint')
    parser.add_argument('--progbar_rate', type=int, default=1, help='How often to update the progress bar in the command line interface')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--num_hidden', type=int, default=512)
    parser.add_argument('--codebook_size', type=int, default=512)
    parser.add_argument('--gumbel', action='store_true')
    args = parser.parse_args()
    # -------------------------------------------------------------------------

    # make sure that relevant dirs exist
    run_name = f'ActionVQVAE/{args.env_name}'
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    # init model
    vqvae_args = {
            'input_dim':64,
            'embedding_dim':args.embedding_dim, 
            'codebook_size':args.codebook_size,
            'log_perplexity':args.log_perplexity,
            'perplexity_freq':args.perplexity_freq,
            'latent_size':args.latent_size,
            'num_hidden':args.num_hidden
        }
    if args.load_from_checkpoint:
        checkpoint_file = os.path.join(log_dir, 'lightning_logs', f'version_{args.version}', 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint_file}')
        model = ActionVQVAE.load_from_checkpoint(checkpoint_file, **vqvae_args)
    else:
        model = ActionVQVAE(**vqvae_args)

    # load data
    data = datasets.BufferedBatchDataset(args.env_name, args.data_dir, args.batch_size, args.epochs)
    train_loader = DataLoader(data, num_workers=args.num_workers)

    # annealing schedules for lots of constants
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='Training/loss', mode='min', save_last=True, every_n_train_steps=args.save_freq))
    # create callbacks to sample reconstructed images
    callbacks.append(DecayLR())
    if args.gumbel:
       callbacks.extend([DecayTemperature(), RampBeta()])
    
    # create trainer instance
    trainer = pl.Trainer(
        callbacks=callbacks, 
        default_root_dir=log_dir, 
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs,
        accelerator='dp',
        log_every_n_steps=args.log_freq,
        progress_bar_refresh_rate=args.progbar_rate
    )
    
    # train model
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
