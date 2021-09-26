"""
Defines the full (PyTorch Lightning module) VQVAE, which incorporates an
encoder, decoder and a quantize layer in the middle for the discrete bottleneck.
"""

import os
import math
from argparse import ArgumentParser, Namespace

import numpy as np
import einops

from torchvision.utils import make_grid

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vqvae_model.deepmind_enc_dec import DeepMindEncoder, DeepMindDecoder
from vqvae_model.openai_enc_dec import OpenAIEncoder, OpenAIDecoder
from vqvae_model.openai_enc_dec import Conv2d as PatchedConv2d
from vqvae_model.quantize import VQVAEQuantize, GumbelQuantize
from vqvae_model.loss import Normal, LogitLaplace

import datasets

# -----------------------------------------------------------------------------

class VQVAE(pl.LightningModule):

    def __init__(self, args, input_channels=3, log_perplexity=False, perplexity_freq=500):
        super().__init__()
        self.save_hyperparameters()

        # encoder/decoder module pair
        Encoder, Decoder = {
            'deepmind': (DeepMindEncoder, DeepMindDecoder),
            'openai': (OpenAIEncoder, OpenAIDecoder),
        }[args.enc_dec_flavor]
        self.encoder = Encoder(input_channels=input_channels, n_hid=args.n_hid)
        self.decoder = Decoder(n_init=args.embedding_dim, n_hid=args.n_hid, output_channels=input_channels)

        # the quantizer module sandwiched between them, +contributes a KL(posterior || prior) loss to ELBO
        QuantizerModule = {
            'vqvae': VQVAEQuantize,
            'gumbel': GumbelQuantize,
        }[args.vq_flavor]
        self.quantizer = QuantizerModule(self.encoder.output_channels, args.num_embeddings, args.embedding_dim)

        # the data reconstruction loss in the ELBO
        ReconLoss = {
            'l2': Normal,
            'logit_laplace': LogitLaplace,
            # todo: add vqgan
        }[args.loss_flavor]
        self.recon_loss = ReconLoss

    def forward(self, x):
        z = self.encoder(self.recon_loss.inmap(x))
        z_q, latent_loss, ind, _ = self.quantizer(z)
        x_hat = self.recon_loss.unmap(self.decoder(z_q))
        return x_hat, latent_loss, ind

    
    @torch.no_grad()
    def reconstruct_only(self, x):
        z = self.encoder(self.recon_loss.inmap(x))
        z_q, *_ = self.quantizer(z)
        x_hat = self.decoder(z_q)
        x_hat = self.recon_loss.unmap(x_hat)
        return x_hat
    
    @torch.no_grad()
    def decode_only(self, z_q):
        x_hat = self.decoder(z_q)
        x_hat = self.recon_loss.unmap(x_hat)
        return x_hat
    
    def decode_with_grad(self, z_q):
        x_hat = self.decoder(z_q)
        x_hat = self.recon_loss.unmap(x_hat)
        return x_hat

    @torch.no_grad()
    def encode_only(self, x):
        z = self.encoder(self.recon_loss.inmap(x))
        z_q, _, ind, neg_dist = self.quantizer(z)
        return z_q, ind, neg_dist
    
    def encode_with_grad(self, x):
        z = self.encoder(self.recon_loss.inmap(x))
        z_q, diff, ind, neg_dist = self.quantizer(z)
        return z_q, diff, ind, neg_dist
    
    def training_step(self, batch, batch_idx):
        # unpack batch and do some basic transforms on the image
        obs, *_ = batch
        img = obs['pov']
        img = einops.rearrange(img, 'b h w c -> b c h w') # switch to channel-first
        img = img.float() / 255 # convert from uint8 to float32

        # forward pass
        img_hat, latent_loss, ind = self.forward(img)
        
        # compute reconstruction loss
        recon_loss = self.recon_loss.nll(img, img_hat)
        
        # loss = reconstruction_loss + codebook loss from quantizer
        loss = recon_loss + latent_loss
        
        # logging
        self.log('Training/loss', loss, on_step=True)
        self.log('Training/recon_loss', recon_loss, on_step=True)
        self.log('Training/latent_loss', latent_loss, on_step=True)
        if self.hparams.log_perplexity:
            if (self.global_step + 1) % self.hparams.perplexity_freq == 0:
                self.eval()
                perplexity, cluster_use = self._compute_perplexity(ind)
                self.train()
                self.log('Training/perplexity', perplexity, prog_bar=True)
                self.log('Training/cluster_use', cluster_use, prog_bar=True)
        return loss

    @torch.no_grad()
    def _compute_perplexity(self, ind):
        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(ind, self.quantizer.n_embed).float().reshape(-1, self.quantizer.n_embed)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        return perplexity, cluster_use

    def configure_optimizers(self):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d, PatchedConv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 1e-4},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        self.optimizer = optimizer

        return optimizer

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


class GenerateCallback(pl.Callback):
    def __init__(self, batch_size=6, dataset=None, save_to_disk=False, every_n_batches=100, precision=32):
        """
        Inputs:
            batch_size - Number of images to generate
            dataset - Dataset to sample from
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_batches = every_n_batches
        self.save_to_disk = save_to_disk
        self.initial_loading = False
        obs, *_ = next(dataset.iter.buffered_batch_iter(batch_size, num_batches=1))
        img = torch.from_numpy(obs['pov'])
        
        img = einops.rearrange(img, 'b h w c -> b c h w')
        img = img.float() / 255
        self.img_batch = img

    def on_batch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (pl_module.global_step+1) % self.every_n_batches == 0:
            self.reconstruct(trainer, pl_module, pl_module.global_step+1)

    def reconstruct(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        if self.img_batch.device != pl_module.device:
            self.img_batch = self.img_batch.to(pl_module.device)

        reconstructed_img = pl_module.reconstruct_only(self.img_batch)

        images = torch.stack([self.img_batch, reconstructed_img], dim=1).reshape((self.batch_size * 2, *self.img_batch.shape[1:]))

        # log images to tensorboard
        trainer.logger.experiment.add_image('Reconstruction',make_grid(images, nrow=2), epoch)


class VisualizeLatents(pl.Callback):
    def __init__(self, every_n_batches=100):
        super().__init__()
        self.every_n_batches = every_n_batches

    def on_batch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (pl_module.global_step+1) % self.every_n_batches == 0:
            self.visualize_latents(trainer, pl_module, pl_module.global_step+1)

    def visualize_latents(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        images = []
        for i in range(pl_module.quantizer.n_embed):
            latent_idcs = torch.ones(1,16*16, dtype=torch.long, device=pl_module.device) * i
            latent = pl_module.quantizer.embed_code(latent_idcs)
            latent = einops.rearrange(latent, 'b (h w) c -> b c h w', h=16, w=16)
            recon = pl_module.decode_only(latent)
            images.append(recon[0].detach().cpu())

        images = torch.stack(images, dim=0)

        # log images to tensorboard
        trainer.logger.experiment.add_image('Latents',make_grid(images, nrow=4), epoch)


def cli_main():
    pl.seed_everything(1337)

    # -------------------------------------------------------------------------
    # arguments...
    parser = ArgumentParser()
    # training related
    parser = pl.Trainer.add_argparse_args(parser)
    # model type
    parser.add_argument("--vq_flavor", type=str, default='gumbel', choices=['vqvae', 'gumbel'])
    parser.add_argument("--enc_dec_flavor", type=str, default='deepmind', choices=['deepmind', 'openai'])
    parser.add_argument("--loss_flavor", type=str, default='l2', choices=['l2', 'logit_laplace'])
    parser.add_argument('--callback_batch_size', type=int, default=6, help='How many images to reconstruct for callback (shown in tensorboard/images)')
    parser.add_argument('--callback_freq', type=int, default=100, help='How often to reconstruct for callback (shown in tensorboard/images)')
    parser.add_argument('--save_freq', type=int, default=500, help='Save the model every N training steps')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--progbar_rate', type=int, default=10)
    # model size
    parser.add_argument("--num_embeddings", type=int, default=32, help="vocabulary size; number of possible discrete states")
    parser.add_argument("--embedding_dim", type=int, default=32, help="size of the vector of the embedding of each discrete token")
    parser.add_argument("--n_hid", type=int, default=64, help="number of channels controlling the size of the model")
# dataloader related
    parser.add_argument("--data_dir", type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument("--env_name", type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument("--batch_size", type=int, default=20)
    # model loading args
    parser.add_argument('--load_from_checkpoint', default=False, action='store_true')
    parser.add_argument('--version', default=None, type=int, help='Version of model, if training is resumed from checkpoint')
    #other args
    parser.add_argument('--log_dir', type=str, default='/home/lieberummaas/datadisk/minerl/run_logs')
    # done!
    args = parser.parse_args()
    # -------------------------------------------------------------------------

    # make sure that relevant dirs exist
    run_name = f'VQVAE/{args.env_name}'
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f'\nSaving logs and model to {log_dir}')

    # init model
    vqvae_args = Namespace(**{
            'vq_flavor':args.vq_flavor, 
            'enc_dec_flavor':args.enc_dec_flavor, 
            'embedding_dim':args.embedding_dim, 
            'n_hid':args.n_hid, 
            'num_embeddings':args.num_embeddings,
            'loss_flavor':args.loss_flavor
        })
    if args.load_from_checkpoint:
        checkpoint_file = os.path.join(log_dir, 'lightning_logs', f'version_{args.version}', 'checkpoints', 'last.ckpt')
        print(f'\nLoading model from {checkpoint_file}')
        model = VQVAE.load_from_checkpoint(checkpoint_file, args=vqvae_args)
    else:
        model = VQVAE(args = vqvae_args)

    # load data
    data = datasets.BufferedBatchDataset(args.env_name, args.data_dir, args.batch_size, num_epochs=1)
    dataloader = DataLoader(data, batch_size=None, num_workers=1)

    # annealing schedules for lots of constants
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='Training/loss', mode='min', save_last=True, every_n_train_steps=args.save_freq))
    # create callbacks to sample reconstructed images
    callbacks.append(
        GenerateCallback(
            batch_size=args.callback_batch_size, 
            dataset=data, 
            save_to_disk=False, 
            every_n_batches=args.callback_freq
        )
    )
    callbacks.append(
        VisualizeLatents(every_n_batches=args.callback_freq)
    )
    callbacks.append(DecayLR())
    if args.vq_flavor == 'gumbel':
       callbacks.extend([DecayTemperature(), RampBeta()])
    
    # create trainer instance
    trainer = pl.Trainer(
        callbacks=callbacks, 
        default_root_dir=log_dir, 
        gpus=torch.cuda.device_count(),
        max_epochs=1,
        accelerator='dp',
        log_every_n_steps=args.log_freq,
        progress_bar_refresh_rate=args.progbar_rate
    )

    trainer.fit(model, dataloader)

if __name__ == "__main__":
    cli_main()
