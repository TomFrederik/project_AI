from copy import deepcopy

import einops
from einops.layers.torch import Rearrange
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from dynamics_models import MDN_RNN
from vae_model import VAE
from vqvae import VQVAE


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
        out = nn.functional.relu(out)
        return out

class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, n_hid=64, latent_dim=1024):
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
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
    
    @torch.no_grad()
    def encode_only(self, x):
        return self(x), None, None
    
    def encode_with_grad(self, x):
        return self(x), 0, None, None
    
    @property
    def device(self):
        return list(self.net.parameters())[0].device

class QNetwork(pl.LightningModule):
    
    def __init__(
        self, 
        n_actions, # number of distinct actions
        optim_kwargs, 
        target_update_rate, # how often to update the target network
        margin, # margin in the classification loss
        discount_factor, 
        horizon, # time horizon for the N-step TD error
        visual_model_cls=VQVAE,
        visual_model_path=None,
        visual_model_kwargs={},
        freeze_visual_model=False,
        dynamics_model_cls=MDN_RNN,
        dynamics_model_path=None,
        freeze_dynamics_model=False,
        use_one_hot=False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # set up feature extractor
        if visual_model_cls in [VQVAE, VAE]:
            self.visual_model = visual_model_cls.load_from_checkpoint(visual_model_path)
            print(f'\nLoaded {visual_model_cls.__name__} from {visual_model_path}!')
            self.visual_model.eval()
        elif visual_model_cls == ConvFeatureExtractor:
            self.visual_model = visual_model_cls(**visual_model_kwargs)
            print('\nInitialized new ConvFeatureExtractor')
        else:
            raise ValueError(f'Unrecognized feature extractor class {visual_model_cls}')
        
        # set up dynamics model
        if dynamics_model_path is not None and dynamics_model_cls is not None:
            self.dynamics_model = dynamics_model_cls.load_from_checkpoint(dynamics_model_path)
            print(f'\nLoaded {dynamics_model_cls.__name__} from {dynamics_model_path}!')
        else:
            print('\nNot using any dynamics model!\n')
            self.dynamics_model = None

        
        # calculate dimension of encoded (equivalent to latent_dim in VAE and latent_dim * num_variables in VQVAE)
        if isinstance(self.visual_model, VQVAE):
            if use_one_hot:
                dummy, *_ = self.visual_model.encode_only_one_hot(torch.ones(2,3,64,64).float().to(self.visual_model.device))
            else:
                dummy, *_ = self.visual_model.encode_only(torch.ones(2,3,64,64).float().to(self.visual_model.device))
            dummy = einops.rearrange(dummy, 'b n d -> b (n d)')
        else:
            dummy, *_ = self.visual_model.encode_only(torch.ones(2,3,64,64).float().to(self.visual_model.device))
        pov_feature_dim = dummy.shape[1]
        print(f'\n{pov_feature_dim = }\n')

        # set up Q-Network
        if self.dynamics_model is not None:
            self.q_net = nn.Sequential(
                nn.Linear(64 + pov_feature_dim + self.dynamics_model.hparams.gru_kwargs['hidden_size'], 1000),
                nn.GELU(),
                nn.Linear(1000, 1000),
                nn.GELU(),
                nn.Linear(1000, self.hparams.n_actions)
            )
        else:
            self.q_net = nn.Sequential(
                nn.Linear(64 + pov_feature_dim, 1000),
                nn.GELU(),
                nn.Linear(1000, 1000),
                nn.GELU(),
                nn.Linear(1000, self.hparams.n_actions)
            )

        # init target net
        self._update_target()
        
        # loss function
        self.loss_fn = nn.MSELoss()
    
    def _update_target(self):
        '''updates target network'''
        self.target_net = nn.ModuleDict({
            'visual_model':deepcopy(self.visual_model),
            'q_net':deepcopy(self.q_net),
        })

        self.target_net.eval()
        
    def forward(self, pov, vec_obs, predictive_state=None, target=False):
        if target:
            # extract pov features
            if self.hparams.use_one_hot and isinstance(self.visual_model, VQVAE):
                pov_out, *_ = self.target_net['visual_model'].encode_only_one_hot(pov)
            else:
                pov_out, *_ = self.target_net['visual_model'].encode_only(pov)

            if isinstance(self.visual_model, VQVAE):
                pov_out = einops.rearrange(pov_out, 'b n d -> b (n d)')
            
            # compute q_values
            if predictive_state is not None:
                q_values = self.target_net['q_net'](torch.cat([pov_out, vec_obs, predictive_state], dim=1))
            else:
                q_values = self.target_net['q_net'](torch.cat([pov_out, vec_obs], dim=1))

            return q_values
        else:
            # extract pov features
            if self.hparams.freeze_visual_model:
                if self.hparams.use_one_hot and isinstance(self.visual_model, VQVAE):
                    pov_out, *_ = self.target_net['visual_model'].encode_only_one_hot(pov)
                else:
                    pov_out, *_ = self.target_net['visual_model'].encode_only(pov)
                codebook_loss = 0
            else:
                raise NotImplementedError()
                #pov_out, codebook_loss, _, _ = self.visual_model.encode_with_grad(pov)
            
            if isinstance(self.visual_model, VQVAE):
                pov_out = einops.rearrange(pov_out, 'b n d -> b (n d)')

            # compute q_values
            if predictive_state is not None:
                q_values = self.q_net(torch.cat([pov_out, vec_obs, predictive_state], dim=1))
            else:
                q_values = self.q_net(torch.cat([pov_out, vec_obs], dim=1))

            return q_values, codebook_loss
    
    def _large_margin_classification_loss(self, q_values, expert_action):
        '''
        Computes the large margin classification loss J_E(Q) from the DQfD paper
        '''
        idcs = torch.arange(0,len(q_values),dtype=torch.long)
        q_values = q_values + self.hparams.margin
        q_values[idcs, expert_action] = q_values[idcs, expert_action] - self.hparams.margin
        return torch.max(q_values, dim=1)[0] - q_values[idcs, expert_action]
    
    def _compute_n_step_rewards(self, rewards):
        '''computes n-step rewards by convolving the discount array with the reward array'''
        discount_array = torch.zeros_like(rewards)[:self.hparams.horizon]
        for i in range(self.hparams.horizon):
            discount_array[i] = self.hparams.discount_factor ** i
    
        n_step_rewards = F.conv1d(rewards[None,None,:], discount_array[None,None,:], padding=self.hparams.horizon)[0,0,:-1]
        n_step_rewards = n_step_rewards[self.hparams.horizon:]

        return n_step_rewards

    def training_step(self, batch, batch_idx):
        one_step_loss, classification_loss, n_step_loss, loss, expert_agent_agreement, expert_q_values, other_q_values, action_idcs = self.step(batch)

        # logging
        log_dict = {
            'Training/1-step TD Error': one_step_loss,
            'Training/ClassificationLoss': classification_loss,
            'Training/n-step TD Error': n_step_loss,
            'Training/Loss': loss,
            'Training/ExpertAgentAgreement': expert_agent_agreement,
            'Training/ExpertQValues': expert_q_values,
            'Training/OtherQValues': other_q_values,
            'Training/Actions': wandb.Histogram(action_idcs.detach().cpu())
        }
        self.logger.experiment.log(log_dict)

        return loss

    def validation_step(self, batch, batch_idx):
        one_step_loss, classification_loss, n_step_loss, loss, expert_agent_agreement, expert_q_values, other_q_values, action_idcs = self.step(batch)

        # logging
        log_dict = {
            'Validation/1-step TD Error': one_step_loss,
            'Validation/ClassificationLoss': classification_loss,
            'Validation/n-step TD Error': n_step_loss,
            'Validation/Loss': loss,
            'Validation/ExpertAgentAgreement': expert_agent_agreement,
            'Validation/ExpertQValues': expert_q_values,
            'Validation/OtherQValues': other_q_values,
            #'Validation/Actions': wandb.Histogram(action_idcs.detach().cpu())
        }
        return log_dict

    def validation_epoch_end(self, outputs):
        log_dict = {}
        for key in outputs[0].keys():
            mean_metric = torch.stack([x[key] for x in outputs], dim=0).mean()
            log_dict[key] = mean_metric
        self.logger.experiment.log(log_dict)


    def step(self, batch): 
        pov_obs, vec_obs, actions, action_idcs, rewards = map(lambda x: x[0], batch) # remove first dimension
        
        # compute n-step rewards
        n_step_rewards = self._compute_n_step_rewards(rewards)

        # 
        if self.dynamics_model is not None:
            if isinstance(self.visual_model, VQVAE):
                if self.hparams.use_one_hot:
                    sample, *_ = self.visual_model.encode_only_one_hot(pov_obs)
                else:
                    sample, *_ = self.visual_model.encode_only(pov_obs) 
                sample = einops.rearrange(sample, 'b d c -> b (d c)')
            else:
                sample, *_ = self.visual_model.encode_only(pov_obs) 
            
            gru_input = torch.cat([sample, vec_obs, actions], dim=1)[None]
            hidden_states_seq, _ = self.dynamics_model.gru(gru_input)
            predictive_state = hidden_states_seq[0]
            predictive_state = torch.cat([torch.zeros_like(predictive_state)[:1], predictive_state[:-1]], dim=0)
    
            # predict q values
            q_values, codebook_loss = self(pov_obs, vec_obs, predictive_state)
            action_idcs = action_idcs
            target_next_q_values = self(pov_obs[1:], vec_obs[1:], predictive_state[1:], target=True)
            next_action = torch.argmax(self(pov_obs[1:], vec_obs[1:], predictive_state[1:])[0], dim=1)
            target_n_step_q_values = self(pov_obs[self.hparams.horizon:], vec_obs[self.hparams.horizon:], predictive_state[self.hparams.horizon:], target=True)
            n_step_action = torch.argmax(self(pov_obs[self.hparams.horizon:], vec_obs[self.hparams.horizon:], predictive_state[self.hparams.horizon:])[0], dim=1)
                
        else:
            # predict q values
            q_values, codebook_loss = self(pov_obs, vec_obs)
            action_idcs = action_idcs
            target_next_q_values = self(pov_obs[1:], vec_obs[1:], target=True)
            next_action = torch.argmax(self(pov_obs[1:], vec_obs[1:])[0], dim=1)
            target_n_step_q_values = self(pov_obs[self.hparams.horizon:], vec_obs[self.hparams.horizon:], target=True)
            n_step_action = torch.argmax(self(pov_obs[self.hparams.horizon:], vec_obs[self.hparams.horizon:])[0], dim=1)
            


        # compute the individual losses
        idcs = torch.arange(0, len(q_values), dtype=torch.long, requires_grad=False)
        classification_loss = self._large_margin_classification_loss(q_values, action_idcs).mean()
        one_step_loss = (q_values[idcs, action_idcs] - rewards - self.hparams.discount_factor * torch.cat([target_next_q_values[idcs[:-1], next_action], torch.zeros_like(rewards)[:1]], dim=0)).pow(2).mean()
        n_step_loss = (q_values[idcs, action_idcs] - n_step_rewards - (self.hparams.discount_factor ** self.hparams.horizon) * torch.cat([target_n_step_q_values[idcs[:-self.hparams.horizon], n_step_action], torch.zeros_like(n_step_rewards)[:self.hparams.horizon]],dim=0)).pow(2).mean()

        # sum up losses
        loss = classification_loss + one_step_loss + n_step_loss + codebook_loss

        # compute perc where expert action has highest q_value for logging
        expert_agent_agreement = (torch.argmax(q_values, dim=1) == action_idcs).sum() / q_values.shape[0]
        
        ## for logging
        expert_q_values = q_values[idcs, action_idcs].mean()
        other_q_values =  deepcopy(q_values.detach())
        other_q_values[idcs, action_idcs] = 0
        other_q_values = other_q_values.mean()
        ##

        return one_step_loss, classification_loss, n_step_loss, loss, expert_agent_agreement, expert_q_values, other_q_values, action_idcs.detach().cpu()


    def on_after_backward(self):
        if (self.global_step + 1) % self.hparams.target_update_rate == 0:
            print(f'\nGlobal step {self.global_step+1}: Updating Target Network\n')
            self._update_target()
        
    def configure_optimizers(self):
        # collect all params
        params = list(self.q_net.parameters())

        # add params of pretrained models if they are not frozen:
        if not self.hparams.freeze_visual_model:
            params += list(self.visual_model.parameters())
        if not self.hparams.freeze_dynamics_model:
            raise NotImplementedError('Dynamics model cant yet be unfrozen --> need to implement!')
            # params += list(self.dynamics_model.parameters())
        
        # set up optimizer
        optimizer = torch.optim.AdamW(params, **self.hparams.optim_kwargs)
        
        return optimizer