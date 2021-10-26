from time import time

import einops
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchdiffeq as teq

from vae_model import VAE
from vqvae import VQVAE


visual_model_by_str = {
    'vae':VAE,
    'vqvae':VQVAE
}

        
class MDN_RNN(pl.LightningModule):
    def __init__(
        self, 
        gru_kwargs, 
        optim_kwargs, 
        #scheduler_kwargs, 
        num_components=5, 
        visual_model_path='', 
        visual_model_cls='vae', 
        # curriculum_threshold=3.0, 
        curriculum_start=0, 
        max_forecast=10,
        use_one_hot=False
    ):
        super().__init__()
        
        # save params
        self.save_hyperparameters()

        # init curriculum # TODO: maybe remove this
        self._init_curriculum(curriculum_start=curriculum_start)
        print(f'\nCurriculum starts at {self.curriculum[self.curriculum_step]} step forecast')
        
        # load VAE
        self.visual_model = visual_model_by_str[visual_model_cls].load_from_checkpoint(visual_model_path)
        self.visual_model.eval() # TODO: maybe make this optional for finetuning --> need to save the new model too!
        
        if self.hparams.visual_model_cls == 'vqvae':
            if use_one_hot:
                print('\nUsing one-hot representation')
                self.latent_dim = self.visual_model.quantizer.num_variables * self.visual_model.quantizer.codebook_size
            else:
                print('\nUsing learned embedding representation')
                self.latent_dim = self.visual_model.quantizer.num_variables * self.visual_model.quantizer.embedding_dim

        elif self.hparams.visual_model_cls == 'vae':
            self.latent_dim = self.visual_model.hparams.encoder_kwargs['latent_dim']
        print(f'\nlatent_dim = {self.latent_dim}')
            
        
        # set up model
        self.gru_input_dim = self.latent_dim + 64 + 64 # 64 for action dim, 64 again for vecobs
        self.gru = nn.GRU(**gru_kwargs, input_size=self.gru_input_dim, batch_first=True)

        if self.hparams.visual_model_cls == 'vqvae':
            self.mdn_network = nn.Sequential(
                nn.Linear(gru_kwargs['hidden_size'], num_components + num_components * self.latent_dim + 64)
            )
        else:
            self.mdn_network = nn.Sequential(
                nn.Linear(gru_kwargs['hidden_size'], num_components + num_components * 2 * self.latent_dim + 64)
            )
        
    def _step(self, batch):
        # unpack batch
        pov, vec, actions, *_ = batch

        # make predictions
        if self.hparams.visual_model_cls == 'vqvae':
            
            logits, mixing_logits, vec_pred, target_probs, target_vec = self(pov, vec, actions)
            target_probs = einops.rearrange(target_probs, 'b t num_vars cb_size -> (b t) (num_vars cb_size)')
            logits = einops.rearrange(logits[:,:-1], 'b t K d -> (b t) K d')
            mixing_logits = einops.rearrange(mixing_logits[:,:-1], 'b t K -> (b t) K')
            vec_pred = vec_pred[:,:-1]
            
            sampled_mix = torch.nn.functional.gumbel_softmax(mixing_logits, tau=1, hard=True, dim=-1)
            sampled_logits = torch.einsum('a b c, a b -> a c', logits, sampled_mix)
            sampled_logits = torch.nn.functional.log_softmax(sampled_logits, dim=1)
            
            pov_loss = -(target_probs * sampled_logits).sum() / self.visual_model.quantizer.num_variables / target_probs.shape[0]

            vec_loss = (target_vec - vec_pred).pow(2).sum(-1).mean()
            
        elif self.hparams.visual_model_cls == 'vae':
            means, log_stds, mixing_logits, vec_pred, target_mean, target_logstd, target_vec = self(pov, vec, actions)

            target_mean = einops.rearrange(target_mean, 'b t d -> (b t) d')
            target_logstd = einops.rearrange(target_logstd, 'b t d -> (b t) d')
            means = einops.rearrange(means[:,:-1], 'b t K d -> (b t) K d')
            log_stds = einops.rearrange(log_stds[:,:-1], 'b t K d -> (b t) K d')
            mixing_logits = einops.rearrange(mixing_logits[:,:-1], 'b t K -> (b t) K')
            vec_pred = vec_pred[:,:-1]

            sample = self.sample_from_gmm(mixing_logits, means, log_stds)
            
            # compute NLL under target dist
            nll = 0.5 * ((target_mean - sample).pow(2) / torch.exp(2* target_logstd)).sum(-1)
            pov_loss = nll.mean()
            vec_loss = (target_vec - vec_pred).pow(2).sum(-1).mean()
        
        return pov_loss, vec_loss
    
    def sample_from_gmm(self, mixing_logits, means, log_stds):

        #sampled_mix = torch.argmax(torch.nn.functional.gumbel_softmax(mixing_logits, tau=1, hard=True, dim=-1),dim=-1)
        # sampled_means = means[torch.arange(len(means)), sampled_mix]
        # sampled_log_stds = log_stds[torch.arange(len(log_stds)), sampled_mix]
        sampled_mix = torch.nn.functional.gumbel_softmax(mixing_logits, tau=1, hard=True, dim=-1)
        sampled_means = torch.einsum('a b c, a b -> a c', means, sampled_mix)
        sampled_log_stds = torch.einsum('a b c, a b -> a c', log_stds, sampled_mix)
        sample = sampled_means + torch.exp(sampled_log_stds) * torch.normal(torch.zeros_like(sampled_means), torch.ones_like(sampled_log_stds))
    
        return sample

    def sample_from_categorical_mixture(self, mixing_logits, logits):

        sampled_mix = torch.nn.functional.gumbel_softmax(mixing_logits, tau=1, hard=True, dim=-1)
        sampled_logits = torch.einsum('a b c, a b -> a c', logits, sampled_mix)
        sampled_logits = einops.rearrange(sampled_logits, 'b (n d) -> b n d', n=32, d=32)
        sampled_one_hot = torch.nn.functional.gumbel_softmax(sampled_logits, hard=True, dim=-1)

        sample = []
        for i in range(sampled_one_hot.shape[1]):
            sample.append(sampled_one_hot[:,i] @ self.visual_model.quantizer.embeds[i].weight)
        sample = torch.stack(sample, dim=1)
        
        return einops.rearrange(sample, 'b n d -> b (n d)')


    def forward(self, pov, vec, actions, last_hidden=None):
        '''
        Given a sequence of pov, vec and actions, computes priors over next latent
        state.
        Inputs:
            pov - (B, T, 3, 64, 64)
            vec - (B, T, 64)
            actions - (B, T, 64)
            last_hidden - (B, gru_kwargs['hidden_size'],), potential last hidden state of the recurrent network
        Output:
            vqvae:
                logits, mixing_logits, vec_pred, target_probs, target_vec
            vae:
                means, log_stds, mixing_logits, vec_pred, target_mean, target_logstd, target_vec
        '''
        # save shape params
        B, T = pov.shape[:2]

        # merge frames with batch for batch processing
        pov = einops.rearrange(pov, 'b t c h w -> (b t) c h w')
        target_vec = vec[:,1:]
        
        # encode pov to latent
        if self.hparams.visual_model_cls == 'vqvae':
            if self.hparams.use_one_hot:
                z_q, probs = self.visual_model.encode_only_one_hot(pov)
                probs = einops.rearrange(probs, '(b t) num_vars cb_size -> b t num_vars cb_size', b=B, t=T)
            else:
                z_q, _, probs = self.visual_model.encode_only(pov)
                probs = einops.rearrange(torch.softmax(probs,dim=2), '(b t) num_vars cb_size -> b t num_vars cb_size', b=B, t=T)

            sample = einops.rearrange(z_q, '(b t) num_vars cb_size -> b t (num_vars cb_size)', b=B, t=T)
            target_probs = probs[:,1:]
            
        elif self.hparams.visual_model_cls == 'vae':
            sample, mean, logstd = self.visual_model.encode_only(pov) 
            sample = einops.rearrange(sample, '(b t) d -> b t d', b=B)
            logstd = einops.rearrange(logstd, '(b t) d -> b t d', b=B)
            mean = einops.rearrange(mean, '(b t) d -> b t d', b=B)

            target_mean = mean[:,1:]
            target_logstd = logstd[:,1:]

        # compute one-step predictions
        input_states = torch.cat([sample, vec], dim=2)
        if self.hparams.visual_model_cls == 'vqvae':
            logits, mixing_logits, vec_pred = self.one_step_prediction(input_states, actions, last_hidden)
            return logits, mixing_logits, vec_pred, target_probs, target_vec
        else:
            means, log_stds, mixing_logits, vec_pred = self.one_step_prediction(input_states, actions, last_hidden)
            return means, log_stds, mixing_logits, vec_pred, target_mean, target_logstd, target_vec
    
    
    def one_step_prediction(self, states, actions, h0=None, log_prior=None):
        '''
        Helper function which takes a sequence of states and the action taken in each state.
        Optionally also takes the last RNN state and prior over next state.
        Computes the belief over the next state
        Input:
            states - (B, T, latent_dim, H, W)
            actions - (B, T, 64)
            h0 - (B, lstm_kwargs['hidden_size'],)
            log_prior
        Output:
            pov_logits or mean - distribution over the next latent space or the mean of the next latent state
            state -  sample from the above distribution
            hidden_state_seq - hidden states of the RNN after each time step
        '''
        # save T for later
        B, T, D = states.shape

        # compute hidden states of gru
        if h0 is None:
            hidden_states_seq, _ = self.gru(torch.cat([states, actions], dim=2))
        else:
            hidden_states_seq, _ = self.gru(torch.cat([states, actions], dim=2), h0)

        # compute next state
        mdn_out = self.mdn_network(einops.rearrange(hidden_states_seq, 'b t d -> (b t) d'))

        if self.hparams.visual_model_cls == 'vqvae':
            mixing_logits = mdn_out[:,:self.hparams.num_components]
            mixing_logits = einops.rearrange(mixing_logits, '(b t) K -> b t K',t=T, b=B)
            logits = torch.chunk(mdn_out[:,self.hparams.num_components:self.hparams.num_components*(1+self.latent_dim)], chunks=self.hparams.num_components, dim=1)
            logits = einops.rearrange(torch.stack(logits, dim=1), '(b t) K d -> b t K d',t=T, b=B)
            vec_pred = mdn_out[:,-64:]
            vec_pred = einops.rearrange(vec_pred, '(b t) d -> b t d', b=B, t=T)
            return logits, mixing_logits, vec_pred
        
        elif self.hparams.visual_model_cls == 'vae':
            # mean == pov_pred
            mixing_logits = mdn_out[:,:self.hparams.num_components]
            mixing_logits = einops.rearrange(mixing_logits, '(b t) K -> b t K',t=T, b=B)
            means = torch.chunk(mdn_out[:,self.hparams.num_components:self.hparams.num_components*(1+self.latent_dim)], chunks=self.hparams.num_components, dim=1)
            means = einops.rearrange(torch.stack(means, dim=1), '(b t) K d -> b t K d',t=T, b=B)
            log_stds = torch.chunk(mdn_out[:,self.hparams.num_components*(1+self.latent_dim):self.hparams.num_components*(1+2*self.latent_dim)], chunks=self.hparams.num_components, dim=1)
            log_stds = einops.rearrange(torch.stack(log_stds, dim=1), '(b t) K d -> b t K d',t=T, b=B)
            vec_pred = mdn_out[:,-64:]
            vec_pred = einops.rearrange(vec_pred, '(b t) d -> b t d', b=B, t=T)

            return means, log_stds, mixing_logits, vec_pred
        
    @torch.no_grad()
    def imaginate(self, starting_pov, starting_vec, action_sequence):
        assert starting_pov.shape == (3, 64, 64), f"{starting_pov.shape = }"
        assert starting_vec.shape == (64,), f"{starting_vec.shape = }"
        assert action_sequence.shape[1:] == (64,), f"{action_sequence.shape[1:] = }"

        # apply frame encoding
        if self.hparams.visual_model_cls == 'vae':
            pov_sample, *_ = self.visual_model.encode_only(starting_pov[None])
        else:
            if self.hparams.use_one_hot:
                pov_sample, *_ = self.visual_model.encode_only_one_hot(starting_pov[None])
                pov_sample = einops.rearrange(pov_sample, 'b num_vars cb_size -> b (num_vars cb_size)')
            else:
                pov_sample, *_ = self.visual_model.encode_only(starting_pov[None])
                pov_sample = einops.rearrange(pov_sample, 'b num_vars emb_dim -> b (num_vars emb_dim)')

        starting_state = torch.cat([pov_sample, starting_vec[None]], dim=1)[None] # 1 1 1088
        #starting_state = pov_sample[None]
        
        # first prediction
        _, h0 = self.gru(torch.cat([starting_state, action_sequence[0][None, None]], dim=-1))

        # predict next state
        mdn_out = self.mdn_network(einops.rearrange(h0, 'b t d -> (b t) d'))
        mixing_logits = mdn_out[:,:self.hparams.num_components]
        vec_pred = mdn_out[:,-64:]

        if self.hparams.visual_model_cls == 'vae':
            means = torch.chunk(mdn_out[:,self.hparams.num_components:self.hparams.num_components*(1+self.latent_dim)], chunks=self.hparams.num_components, dim=1)
            means = torch.stack(means, dim=1)
            log_stds = torch.chunk(mdn_out[:,self.hparams.num_components*(1+self.latent_dim):self.hparams.num_components*(1+2*self.latent_dim)], chunks=self.hparams.num_components, dim=1)
            log_stds = torch.stack(log_stds, dim=1)
            sample = self.sample_from_gmm(mixing_logits, means, log_stds)
        else:
            logits = torch.chunk(mdn_out[:,self.hparams.num_components:self.hparams.num_components*(1+self.latent_dim)], chunks=self.hparams.num_components, dim=1)
            logits = torch.stack(logits, dim=1)
            sample = self.sample_from_categorical_mixture(mixing_logits, logits)

        
        sample = torch.cat([sample, vec_pred], dim=-1)
        sample_list = [sample]

        for t in range(len(action_sequence)-1):
            action = action_sequence[t+1]
            gru_input = torch.cat([sample[None], action[None,None]], dim=-1)
            _, h0 = self.gru(gru_input, h0)

            # predict next state
            mdn_out = self.mdn_network(einops.rearrange(h0, 'b t d -> (b t) d'))
            mixing_logits = mdn_out[:,:self.hparams.num_components]
            vec_pred = mdn_out[:,-64:]
            if self.hparams.visual_model_cls == 'vae':
                means = torch.chunk(mdn_out[:,self.hparams.num_components:self.hparams.num_components*(1+self.latent_dim)], chunks=self.hparams.num_components, dim=1)
                means = torch.stack(means, dim=1)
                log_stds = torch.chunk(mdn_out[:,self.hparams.num_components*(1+self.latent_dim):self.hparams.num_components*(1+2*self.latent_dim)], chunks=self.hparams.num_components, dim=1)
                log_stds = torch.stack(log_stds, dim=1)
                sample = self.sample_from_gmm(mixing_logits, means, log_stds)
            else:
                logits = torch.chunk(mdn_out[:,self.hparams.num_components:self.hparams.num_components*(1+self.latent_dim)], chunks=self.hparams.num_components, dim=1)
                logits = torch.stack(logits, dim=1)
                sample = self.sample_from_categorical_mixture(mixing_logits, logits)
            sample = torch.cat([sample, vec_pred], dim=-1)
            sample_list.append(sample)
        return torch.stack(sample_list, dim=1)[0]

    def training_step(self, batch, batch_idx):
        # perform predictions and compute loss
        pov_loss, vec_loss = self._step(batch)
        loss = pov_loss + vec_loss
        
        # score and log predictions
        self.log('Training/loss', loss,)
        self.log('Training/pov_loss',pov_loss)
        self.log('Training/vec_loss',vec_loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # perform predictions and compute loss
        pov_loss, vec_loss = self._step(batch)
        loss = pov_loss + vec_loss
        
        # score and log predictions
        self.log('Validation/loss', loss,)
        self.log('Validation/pov_loss',pov_loss)
        self.log('Validation/vec_loss',vec_loss)

        return loss
        
    def validation_epoch_end(self, batch_losses):
        # check whether to go to next step in curriculum, 
        # but only if latent overshooting is active
        self._check_curriculum_cond()
    
    def configure_optimizers(self):
        # set up optimizer
        params = list(self.gru.parameters()) + list(self.mdn_network.parameters())
        optimizer = torch.optim.AdamW(params, **self.hparams.optim_kwargs, weight_decay=0)
        return optimizer

    def _init_curriculum(self, max_forecast=None, curriculum_start=0):
        self.curriculum_step = 0
        if max_forecast is None:
            max_forecast = self.hparams.max_forecast
        self.curriculum = [i for i in range(max_forecast-2)]
        self.curriculum_step = curriculum_start
        
    def _check_curriculum_cond(self):
        if self.curriculum_step < len(self.curriculum)-1:
            # if value < self.hparams.curriculum_threshold:
                # self.curriculum_step += 1
                # print(f'\nCurriculum updated! New forecast horizon is {self.curriculum[self.curriculum_step]}\n')
            if (self.current_epoch + 1) % 10 == 0:
                self.curriculum_step += 1
                print(f'\nCurriculum updated! New forecast horizon is {self.curriculum[self.curriculum_step]}\n')
        

