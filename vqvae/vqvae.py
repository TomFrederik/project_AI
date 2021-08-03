"""
Defines the full (PyTorch Lightning module) VQVAE, which incorporates an
encoder, decoder and a quantize layer in the middle for the discrete bottleneck.
"""

import os
import math
from argparse import ArgumentParser

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.deepmind_enc_dec import DeepMindEncoder, DeepMindDecoder
from model.openai_enc_dec import OpenAIEncoder, OpenAIDecoder
from model.openai_enc_dec import Conv2d as PatchedConv2d
from model.quantize import VQVAEQuantize, GumbelQuantize
from model.loss import Normal, LogitLaplace

import datasets

# -----------------------------------------------------------------------------

class VQVAE(pl.LightningModule):

    def __init__(self, args, input_channels=3):
        super().__init__()
        self.args = args

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
        z = self.encoder(x)
        z_q, latent_loss, ind = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, ind
    
    @torch.no_grad()
    def reconstruct_only(self, x):
        z = self.encoder(x)
        z_q, _, _ = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_ha
    

    def training_step(self, batch, batch_idx):
        img = batch[1]
        img = self.recon_loss.inmap(img)
        img_hat, latent_loss, ind = self.forward(img)
        recon_loss = self.recon_loss.nll(img, img_hat)
        loss = recon_loss + latent_loss
        self.log('Training/batch_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch[1]
        img = self.recon_loss.inmap(img)
        img_hat, latent_loss, ind = self.forward(img)
        recon_loss = self.recon_loss.nll(img, img_hat)
        loss = recon_loss + latent_loss
        self.log('Validation/loss', loss, on_epoch=True)
        return loss

        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(ind, self.quantizer.n_embed).float().reshape(-1, self.quantizer.n_embed)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        self.log('val_perplexity', perplexity, prog_bar=True)
        self.log('val_cluster_use', cluster_use, prog_bar=True)

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # model type
        parser.add_argument("--vq_flavor", type=str, default='vqvae', choices=['vqvae', 'gumbel'])
        parser.add_argument("--enc_dec_flavor", type=str, default='deepmind', choices=['deepmind', 'openai'])
        parser.add_argument("--loss_flavor", type=str, default='l2', choices=['l2', 'logit_laplace'])
        # model size
        parser.add_argument("--num_embeddings", type=int, default=512, help="vocabulary size; number of possible discrete states")
        parser.add_argument("--embedding_dim", type=int, default=192, help="size of the vector of the embedding of each discrete token")
        parser.add_argument("--n_hid", type=int, default=64, help="number of channels controlling the size of the model")
        return parser

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

def cli_main():
    pl.seed_everything(1337)

    # -------------------------------------------------------------------------
    # arguments...
    parser = ArgumentParser()
    # training related
    parser = pl.Trainer.add_argparse_args(parser)
    # model related
    parser = VQVAE.add_model_specific_args(parser)
    # dataloader related
    parser.add_argument("--data_dir", type=str, default='/home/lieberummaas/datadisk/minerl/data/numpy_data')
    parser.add_argument("--env_name", type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=6)
    # model loading args
    parser.add_argument('--load_from_checkpoint', default=False, action='store_true')
    parser.add_argument('--version', default=None, type=int, help='Version of model, if training is resumed from checkpoint')
    #other args
    parser.add_argument('--log_dir')
    # done!
    args = parser.parse_args()
    # -------------------------------------------------------------------------

    # make sure that relevant dirs exist
    run_name = f'VQVAE/{args.env_name}'
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f'Saving logs and model to {args.log_dir}')

    # init model
    if args.loading_from_checkpoint:
        checkpoint_file = os.path.join(args.log_dir, run_name, 'lightning_logs', str(args.version), 'checkpoints', 'last.ckpt')
        print(f'Loading model from {checkpoint_file}')
        model = VQVAE.load_from_checkpoint(checkpoint_file)
    else:
        model = VQVAE(args)

    # load data
    data = datasets.VAEData(env_name, data_dir, num_data)
    lengths = [len(data)-int(len(data)*val_perc), int(len(data)*val_perc)]
    train_data, val_data = random_split(data, lengths)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)

    num_batches = len(train_data) // batch_size
    if len(train_data) % batch_size != 0:
        num_batches += 1

    print(f'\nnum train samples = {len(train_data)} --> {num_batches} train batches')
    print(f'num val samples = {len(val_data)}')

    # annealing schedules for lots of constants
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='Validation/loss', mode='min'))
    # create callbacks to sample reconstructed images
    callbacks.append(GenerateCallback(dataset=val_data, save_to_disk=False, precision=precision))
    callbacks.append(DecayLR())
    if args.vq_flavor == 'gumbel':
       callbacks.extend([DecayTemperature(), RampBeta()])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    cli_main()
