import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import torchdiffeq as teq
import einops

import visual_models
import util_models
import vqvae
from reward_model import RewardMLP

from time import time

vae_model_by_str = {
    'vae':visual_models.VAE,
    'vqvae':vqvae.VQVAE
}

EPS = 1e-10

plt.switch_backend('agg')

class MDNRNNReward(nn.Module):
    def __init__(self, mdn_path, reward_path):
        super().__init__()
        self.mdn = MDN_RNN.load_from_checkpoint(mdn_path)
        self.reward_model = RewardMLP.load_from_checkpoint(reward_path)
    
    def forward(self, state, action, h_n, c_n, batched=True):
        _, state, (h_n, c_n), _, _ = self.mdn.forward_latent(state, action, h_n, c_n, batched)
        rew = self.reward_model(state[...,-64:])
        return state, rew, (h_n, c_n)
        
class MDN_RNN(pl.LightningModule):
    def __init__(
        self, 
        gru_kwargs, 
        optim_kwargs, 
        scheduler_kwargs, 
        seq_len, 
        num_components, 
        VAE_path, 
        VAE_class='vae', 
        latent_overshooting=False,
        temp=1, 
        conditioning_len=0,
        curriculum_threshold=3.0, 
        curriculum_start=0, 
        predict_idcs_directly=False, 
        embed=False
    ):
        super().__init__()
        
        # save params
        self.save_hyperparameters()

        # init curriculum
        self._init_curriculum(curriculum_start=curriculum_start)
        print(f'\nCurriculum starts at {self.curriculum[self.curriculum_step]} step forecast')
        
        # load VAE
        self.VAE = vae_model_by_str[VAE_class].load_from_checkpoint(VAE_path)
        #self.VAE.eval()
        
        if self.hparams.VAE_class == 'vqvae':
            self.latent_dim = self.VAE.hparams.args.embedding_dim
            self.num_embeddings = self.VAE.hparams.args.num_embeddings
            print(f'\nlatent_dim = {self.latent_dim}')
            print(f'\nnum_embeddings = {self.num_embeddings}')
            if self.hparams.predict_idcs_directly and self.hparams.embed:
                # init embedding to the vectors themselves
                self.embedding = nn.Embedding.from_pretrained(
                        embeddings = torch.cat([
                            self.VAE.quantizer.embed.weight, 
                            torch.zeros_like(self.VAE.quantizer.embed.weight)],dim=1), 
                        freeze = False
                )
                # init the new part of the embedding with N(0,1)
                nn.init.normal_(self.embedding.weight[:,self.latent_dim:], mean=0.0, std=1.0)
                #self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.latent_dim)

            dummy, dummy_idcs, _ = self.VAE.encode_only(torch.ones(2,3,64,64).float().to(self.VAE.device))
            if self.hparams.predict_idcs_directly and self.hparams.embed:
                dummy = einops.rearrange(self.embedding(dummy_idcs), 'b h w D -> b D h w')
            self.latent_h = dummy.shape[-1]
            self.latent_size = np.prod(dummy.shape[2:])
            
            conv_net = True
            if conv_net:
                num_channels = dummy.shape[1]
                self.conv_net = nn.Sequential(
                    nn.Conv2d(in_channels=num_channels, out_channels=256, kernel_size=3, padding=1, stride=2), # 16 -> 8
                    nn.GELU(),
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2), # 8 -> 4
                    nn.GELU(),
                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2), # 4 -> 2
                    nn.GELU(),
                    nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, stride=2)#, # 2 -> 1
                    #nn.GELU()
                )
                print('\nUsing conv net')
                dummy = self.conv_net(dummy)
                self.pre_gru_size = np.prod(dummy.shape[2:])
                self.pre_gru_channels = dummy.shape[1]
                print(f'\npre_gru_size (H*W) = {self.pre_gru_size}')
            
            else:
                print('\nNot using conv net')
                self.conv_net = None
                self.pre_gru_channels = self.latent_size
                self.pre_gru_size = self.latent_dim
                
            print(f'\nlatent_size (H*W) = {self.latent_size}')
    
        elif self.hparams.VAE_class == 'vae':
            self.latent_dim = self.VAE.hparams.encoder_kwargs['latent_dim']
            print(f'\nlatent_dim = {self.latent_dim}')

            dummy_mean, dummy_std, dummy_sample = self.VAE.encode_only(torch.ones(2,3,64,64).float().to(self.VAE.device))
            print(f'{dummy_sample.shape = }')
            self.latent_h = dummy_sample.shape[-1]
            self.latent_size = np.prod(dummy_sample.shape[2:])
            
            conv_net = True
            if conv_net:
                num_channels = dummy_sample.shape[1]
                self.conv_net = nn.Sequential(
                    nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, padding=1, stride=2), # 16 -> 8
                    nn.GELU(),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2), # 8 -> 4
                    nn.GELU(),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2), # 4 -> 2
                    nn.GELU(),
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)#, # 2 -> 1
                )
                print('\nUsing conv net')
                dummy = self.conv_net(dummy_sample)
                self.pre_gru_size = np.prod(dummy.shape[2:])
                self.pre_gru_channels = dummy.shape[1]
                print(f'\npre_gru_size (H*W) = {self.pre_gru_size}')
        
        '''
        dummy_sample.requires_grad = False
        print(dummy_sample.requires_grad)
        dummy = self.conv_net(dummy_sample)
        loss = dummy.sum()
        loss.backward()
        print(self.conv_net[2].weight.grad[0,0,0,0])
        raise ValueError
        '''
        # set up model
        gru_input_dim = self.pre_gru_channels * self.pre_gru_size + 64
        self.gru = nn.GRU(**gru_kwargs, input_size=gru_input_dim, batch_first=True)
        
        if self.hparams.VAE_class == 'vqvae':
            self.mdn_network = nn.Sequential(
                nn.ConvTranspose2d(in_channels=gru_kwargs['hidden_size'], out_channels=1024, kernel_size=3, padding=1, stride=2, output_padding=1), # 1 -> 2
                nn.GELU(),
                nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, stride=2, output_padding=1), # 2 -> 4
                nn.GELU(),
                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1), # 4 -> 8
                nn.GELU(),
                nn.ConvTranspose2d(
                    in_channels=256, out_channels=self.num_embeddings if self.hparams.predict_idcs_directly else self.latent_dim, 
                    kernel_size=3, padding=1, stride=2, output_padding=1
                    )  # 8 -> 16
            )
            
        else:
            self.mdn_network = nn.Sequential(
                nn.ConvTranspose2d(in_channels=gru_kwargs['hidden_size'], out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1), # 1 -> 2
                nn.GELU(),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2, output_padding=1), # 2 -> 4
                nn.GELU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1), # 4 -> 8
                nn.GELU(),
                nn.ConvTranspose2d(
                    in_channels=64, out_channels=self.latent_dim, 
                    kernel_size=3, padding=1, stride=2, output_padding=1
                    )  # 8 -> 16
            )

    def _step(self, batch):
        # unpack batch
        pov, vec, actions, _ = batch

        # make predictions
        if self.hparams.VAE_class == 'vqvae':
            pov_logits_list, pov_sample_list, target = self(pov, vec, actions, latent_overshooting=self.hparams.latent_overshooting)

            # Compute loss over all time horizons
            loss = 0
            T = self.hparams.seq_len-1
            loss_list = []
            entropy_list = []
            for t in range(T):
                cur_pov_target = target['pov'][:,t]

                pov_logits = pov_logits_list[t]
                pov_logits = einops.rearrange(pov_logits, 'b t num_embed latent_size -> b num_embed latent_size t')#, latent_size=self.latent_size)
                # target is the ground truth state at time t, which should be compared to prediction which started from time t

                cur_pov_target = einops.repeat(cur_pov_target, 'b latent_size -> b latent_size t', t=pov_logits.shape[-1])
                cur_loss = nn.CrossEntropyLoss()(pov_logits, cur_pov_target)

                # compute entropy for logging
                #probs = torch.softmax(-pov_dist, dim=1)
                #entropy = - (probs * torch.log(probs)).sum(dim=1).mean()
                entropy = 0
                loss += cur_loss/T
                loss_list.append(cur_loss)
                entropy_list.append(entropy)
                #print(f'loss.shape = {loss.shape}')
        
        elif self.hparams.VAE_class == 'vae':
            pov_mean_list, pov_sample_list, target = self(pov, vec, actions, latent_overshooting=self.hparams.latent_overshooting)
        
            # Compute loss over all time horizons
            loss = 0
            T = self.hparams.seq_len-1
            loss_list = []
            entropy_list = []
            for t in range(T):
                cur_pov_target = target['pov'][:,t]

                pov_mean = pov_mean_list[t]
                # target is the ground truth state at time t, which should be compared to prediction which started from time t
                pov_mean = einops.rearrange(pov_mean, 'b t c h w -> b c (h w) t')
                cur_target_mean = einops.repeat(cur_pov_target, 'b c hw -> b c hw t', t=pov_mean.shape[-1])
                #print(f'{pov_mean[0] = }')
                #print(f'{cur_target_mean[0] = }')
                # cheat for now
                #print(f'{pov_log_std[0] = }')
                #print(f'{cur_target_log_std[0] = }')
                # compute KL divergence
                cur_loss = (pov_mean - cur_target_mean).pow(2).mean()

                # compute entropy for logging
                #probs = torch.softmax(-pov_dist, dim=1)
                #entropy = - (probs * torch.log(probs)).sum(dim=1).mean()
                entropy = 0
                loss += cur_loss/T
                loss_list.append(cur_loss)
                entropy_list.append(entropy)
                #print(f'loss.shape = {loss.shape}')
        
        return loss, loss_list, entropy_list
    '''
    def on_after_backward(self):
        print(f'{self.conv_net[2].weight.grad.pow(2).sum() = }')
        print(f'{self.gru.weight_hh_l0.grad.pow(2).sum() = }')
        print(f'{self.mdn_network[0].weight.grad.pow(2).sum() = }')
    '''

    def forward(self, pov, vec, actions, h0=None, latent_overshooting=False):
        '''
        Given a sequence of pov, vec and actions, computes priors over next latent
        state.
        Inputs:
            pov - ([B], T, 3, 64, 64)
            vec - ([B], T, 64)
            actions - ([B], T, 64)
            h0 - ([B], lstm_kwargs['hidden_size'],)
            latent_overshooting - Bool, whether to use latent overshooting
        Output:
            pov_dist_list - List, pov_dist_list[t] contains predictions of the pov distribution
                            at time t, starting from different states previous to t
            target - ([B], T-1, latent_dim + vec_dim) sample of ground truth encoding
        '''
        
        last_hidden = h0

        # merge frames with batch
        b = pov.shape[0]
        seq_len = pov.shape[1]
        pov = einops.rearrange(pov, 'b t c h w -> (b t) c h w')
        
        # encode pov to latent
        if self.hparams.VAE_class == 'vqvae':
            #print(f'pov.shape = {pov.shape}')
            pov_sample, ind, log_priors = self.VAE.encode_only(pov)
            if self.hparams.embed:
                pov_sample = einops.rearrange(self.embedding(ind), 'bt h w D -> bt D h w')
            log_priors = einops.rearrange(log_priors, '(b t) D h w -> b t D h w', t=self.hparams.seq_len)
            #print(f'log_priors.shape = {log_priors.shape}')
            one_step_priors = einops.rearrange(log_priors[:,:-1], 'b t D h w -> (b t h w) D')    
            ind = einops.rearrange(ind, 'bt h w -> bt (h w)')
            ind = einops.rearrange(ind, '(b t) hw -> b t hw', b=b)[:,self.hparams.conditioning_len:]
        elif self.hparams.VAE_class == 'vae':
            # to do it like in paper, we just use a sample as target
            mean, _, pov_sample = self.VAE.encode_only(pov) 

        pov_sample = einops.rearrange(pov_sample, '(b t) c h w -> b t c h w', t=self.hparams.seq_len+self.hparams.conditioning_len)
        states = einops.rearrange(pov_sample[:,self.hparams.conditioning_len:], 'b t c h w -> b t c (h w)')
        actions = actions[:,self.hparams.conditioning_len:]
        
        # stack pov and vec to construct target
        if self.hparams.VAE_class == 'vqvae':
            target = {
                'pov': ind[:,1:]
                #'vec': vec[:,1:]
            }
        elif self.hparams.VAE_class == 'vae':
            target_mean = einops.rearrange(mean, '(b t) c h w -> b t c (h w)', b=b)[:,1:]
            target = {
                'pov': target_mean
            }
            
        # condition on previous sequence to prime the RNN
        if self.hparams.conditioning_len > 0:
            if self.hparams.predict_idcs_directly:
                raise NotImplementedError
            if self.conv_net is None:
                raise NotImplementedError
            else:
                condition_states = pov_sample[:,:self.hparams.conditioning_len]
                condition_states = self.conv_net(einops.rearrange(condition_states, 'b t c h w -> (b t) c h w'))
                condition_states = einops.rearrange(condition_states, '(b t) c h w -> b t (c h w)', t=self.hparams.conditioning_len)
                condition_actions = actions[:,:self.hparams.conditioning_len]
                gru_input = torch.cat([condition_states, condition_actions], dim=2)
                if last_hidden is None:
                    hidden_seq, last_hidden = self.gru(gru_input)
                else:
                    hidden_seq, last_hidden = self.gru(gru_input, last_hidden)

        pov_logits_list = []
        pov_mean_list = []
        pov_log_std_list = []
        pov_sample_list = []
        
        # compute one-step predictions
        if self.hparams.VAE_class == 'vqvae':
            pov_logits, sample, hidden_seq = self.one_step_prediction(states[:,:-1], actions[:,:-1], last_hidden, one_step_priors)
            pov_logits_list.extend([[pov_logits[:,i]] for i in range(pov_logits.shape[1])])
        else:
            mean_prior = einops.rearrange(mean, '(b t) c h w -> b t c h w', b=b)[:,self.hparams.conditioning_len:-1]
            states = einops.rearrange(mean_prior, 'b t c h w -> b t c (h w)')
            mean, sample, hidden_seq = self.one_step_prediction(states, actions[:,:-1], last_hidden, mean_prior)
            pov_mean_list.extend([[mean[:,i]] for i in range(mean.shape[1])])
        
        # save sample to list
        pov_sample_list.extend([[sample[:,i]] for i in range(sample.shape[1])])

        
        # latent overshooting
        if latent_overshooting and self.curriculum[self.curriculum_step] > 0:
            # extrapolate/imagine from each state
            for t in range(self.hparams.seq_len-2):
                time1 = time()
                cur_actions = actions[:,1+t:]
                starting_state = states[:,1+t]
                h_0 = hidden_seq[:,t,:]
                
                if self.hparams.VAE_class == 'vqvae':
                    log_prior = einops.rearrange(log_priors[:,1+t], 'b D h w -> (b h w) D')
                    extrapolated_states, extrapolated_pov_logits = self.extrapolate_latent(starting_state, cur_actions, h_0, log_prior)
                    # save results to lists
                    for i in range(extrapolated_states.shape[1]):
                        pov_logits_list[t+1+i].append(extrapolated_pov_logits[:,i])
                        pov_sample_list[t+1+i].append(extrapolated_states[:,i])

                elif self.hparams.VAE_class == 'vae':
                    #TODO log_prior = (mean, log_std)
                    raise NotImplementedError
                    extrapolated_states, (extrapolated_pov_mean, extrapolated_pov_log_std) = self.extrapolate_latent(starting_state, cur_actions, h_0, log_prior)
                    # save results to lists
                    for i in range(extrapolated_states.shape[1]):
                        pov_mean_list[t+1+i].append(extrapolated_pov_mean[:,i])
                        pov_sample_list[t+1+i].append(extrapolated_states[:,i])

                print(f'Time in extrapolate latent, t={t}: {time()-time1:.3f}s')
        
        # stack lists into tensors
        if self.hparams.VAE_class == 'vqvae':
            for i in range(len(pov_sample_list)):
                pov_logits_list[i] = torch.stack(pov_logits_list[i], dim=1)
                pov_sample_list[i] = torch.stack(pov_sample_list[i], dim=1)
            
            return pov_logits_list, pov_sample_list, target
        elif self.hparams.VAE_class == 'vae':
            for i in range(len(pov_sample_list)):
                pov_mean_list[i] = torch.stack(pov_mean_list[i], dim=1)
                pov_sample_list[i] = torch.stack(pov_sample_list[i], dim=1)
            
            return pov_mean_list, pov_sample_list, target
            
    def one_step_prediction(self, states, actions, h0=None, log_prior=None):
        '''
        Helper function which takes (a sample of the current belief over the) current state or a sequence thereof,
        the action taken in that state or states, as well as the current lstm state and computes a belief
        over the next state
        Input:
            states - ([B], T, 64 + latent_dim)
            actions - ([B], T, 64)
            h0 - ([B], lstm_kwargs['hidden_size'],)
            log_prior
        Output:
            pov_dist - distribution over the next latent space
            s_t -  sample from the above distribution
            (h_t, c_t) - hidden and cell state of the lstm after each time step (if stepwise, otherwise only after the last step)
        '''
        # concat states and action
        seq_len = states.shape[1]
        if self.conv_net is not None:
            new_states = self.conv_net(einops.rearrange(states, 'b t c (h w) -> (b t) c h w', h=self.latent_h))
            new_states = einops.rearrange(new_states, '(b t) c h w -> b t (c h w)', t=seq_len)
        else:
            new_states = einops.rearrange(states, 'b t c hw -> b t (c hw)')
        # compute hidden states of gru
        if h0 is None:
            hidden_states_seq, _ = self.gru(torch.cat([new_states, actions], dim=2))
        else:
            hidden_states_seq, _ = self.gru(torch.cat([new_states, actions], dim=2), h0)


        # compute next state
        pov_pred = self.mdn_network(einops.rearrange(hidden_states_seq, 'b t d -> (b t) d 1 1'))
        
        '''
        if self.global_step == 1:
            print(pov_pred.requires_grad)
            loss = pov_pred.pow(2).sum()
            print(f'{loss.item() = }')
            loss.backward()
            print(f'{self.conv_net[2].weight.grad[0,0,0,0] = }')
            print(f'{self.gru.weight_hh_l0.grad[0,0] = }')
            print(f'{self.mdn_network[0].weight.grad[0,0,0] = }')
            raise ValueError
        '''

        if self.hparams.VAE_class == 'vqvae':
            pov_logits = einops.rearrange(pov_pred, 'bt embedding_dim h w -> (bt h w) embedding_dim') # embedding_dim or num_embeddings
            # skip connection / update priors over discrete embedding vectors
            if log_prior is not None:
                pov_logits += log_prior
            
            # sample next state
            one_hot_ind = nn.functional.gumbel_softmax(pov_logits, dim=-1, tau=self.hparams.temp, hard=True)
            state = self.VAE.quantizer.embed_one_hot(one_hot_ind)
            state = einops.rearrange(state, '(b t latent_size) embed_dim -> b t embed_dim latent_size', latent_size=self.latent_size, t=seq_len)
        
            pov_logits = einops.rearrange(pov_logits, '(b t latent_size) num_embeds -> b t num_embeds latent_size', t=seq_len, latent_size=self.latent_size)
            return pov_logits, state, hidden_states_seq
        
        elif self.hparams.VAE_class == 'vae':
            #TODO: currently only implicitly using the prior mean and std in the mdn-rnn, but not updating/skip connection
            
            # generate mean and log_std from mdn-rnn output
            mean = pov_pred

            mean = einops.rearrange(mean, '(b t) D h w -> b t D h w', t=seq_len)
            #print(f'{mean[0,0,0,0] = }')

            # skip connection for mean
            #mean = mean + log_prior

            # sample state
            state = mean + torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
            
            return mean, state, hidden_states_seq

    def extrapolate_latent(self, state, actions, h0=None, log_prior=None):
        '''
        Extrapolate from starting state conditional on actions and pre-existing lstm states
        Args:
            state - ([B], 64 + latent_dim)
            actions - ([B], T', 64), where T' is the number of steps the function will extrapolate
        Returns:
            extrapolated_states - ([B], T', num_components * (64 + latent_dim))
            extrapolated_pov_dist - ([B], T', num_components * (64 + latent_dim))
        '''
        if h0 is not None:
            h_n = h0[None,:].contiguous()
        else:
            h_n = h0
        #print(f'h_n.shape = {h_n.shape}')

        steps = min(self.curriculum[self.curriculum_step], actions.shape[1]-1)
        b = state.shape[0]
        
        extrapolated_states = []
        extrapolated_pov_logits = []
        extrapolated_pov_means = []
        extrapolated_pov_log_stds = []
                
        for t in range(steps):
            if self.conv_net is not None:
                new_state = self.conv_net(einops.rearrange(state, 'b c (h w) -> b c h w', h=self.latent_h))
                new_state = einops.rearrange(new_state, 'b c h w -> b (c h w)')

            if t == 0 and self.conv_net is None:
                new_state = einops.rearrange(new_state, 'b embed_dim latent_size -> b (embed_dim latent_size)')
            
            _, h_n = self.gru(torch.cat([new_state, actions[:,t]], dim=1)[:,None], h_n)
            
            # compute next state
            pov_pred = self.mdn_network(einops.rearrange(h_n[0], 'b d -> b d 1 1'))

            pov_pred = einops.rearrange(pov_pred, '(b t) embedding_dim h w -> b t embedding_dim (h w)', t=1) # embedding_dim or num_embeddings
            if self.hparams.VAE_class == 'vqvae':
                pov_logits = einops.rearrange(pov_pred, 'bt embedding_dim h w -> (bt h w) embedding_dim') # embedding_dim or num_embeddings
                # skip connection / update priors over discrete embedding vectors
                if log_prior is not None:
                    pov_logits += log_prior
                one_hot_ind = nn.functional.gumbel_softmax(pov_logits, dim=-1, tau=self.hparams.temp, hard=True)
                state = self.VAE.quantizer.embed_one_hot(one_hot_ind)
            
                pov_logits = einops.rearrange(pov_logits, '(b latent_size) num_embeds -> b num_embeds latent_size', latent_size=self.latent_size)
                state = einops.rearrange(state, '(b latent_size) embed_dim -> b embed_dim latent_size', latent_size=self.latent_size)
                
                if log_prior is not None:
                    log_prior = einops.rearrange(pov_logits, 'b num_embeds latent_size -> (b latent_size) num_embeds')
            
                if self.hparams.predict_idcs_directly and self.hparams.embed:
                    state = one_hot_ind @ self.embedding.weight
                    state = einops.rearrange(state, '(b latent_size) embed_dim -> b embed_dim latent_size', latent_size=self.latent_size)

                extrapolated_pov_logits.append(pov_logits)
            elif self.hparams.VAE_class == 'vae':
                #TODO: currently only implicitly using the prior mean and std in the mdn-rnn, but not updating/skip connection
                
                # generate mean and log_std from mdn-rnn output
                mean, log_std = torch.chunk(pov_pred, chunks=2, dim=1)

                # skip connection for mean
                mean += log_prior[0]

                # sample state
                state = mean + torch.exp(log_std) * torch.normal(torch.zeros_like(mean), torch.ones_like(log_std))
                
                extrapolated_pov_means.append(mean)
                extrapolated_pov_log_stds.append(log_std)
            
            extrapolated_states.append(state)
        
        # stack the outputs and return
        extrapolated_states = torch.stack(extrapolated_states, dim=1)
        if self.hparams.VAE_class == 'vqvae':
            extrapolated_pov_logits = torch.stack(extrapolated_pov_logits, dim=1)    
            return extrapolated_states, extrapolated_pov_logits
        
        elif self.hparams.VAE_class == 'vae':
            extrapolated_pov_log_stds = torch.stack(extrapolated_pov_log_stds, dim=1)
            extrapolated_pov_means = torch.stack(extrapolated_pov_means, dim=1)
            return extrapolated_states, (extrapolated_pov_means, extrapolated_pov_log_stds)
        
    @torch.no_grad()
    def predict_recursively(self, states, actions, horizon, log_priors):
        '''
        Auto-regressively applies dynamics model. Actions for imagination are supplied, so only states are being predicted
        Input:
            states - (T, D), where D is latent_dim + obf_vector_dim
            actions - (T + H, D_a), where D_a is obf_action_dim and H is the horizon
            horizon - int, number of time steps to extrapolate
        Output:
            predicted_states - (H, D)
        '''
        assert horizon > 0, f"horizon must be greater 0, but is {horizon}!"
        
        
        print('\nSetting curriculum to max!\n')
        seq_len = states.shape[0]
        self._init_curriculum(seq_len)
        self.curriculum_step = len(self.curriculum) - 1
        h = states.shape[2]
        if self.hparams.embed:
            raise NotImplementedError
        
        one_step_priors = einops.rearrange(log_priors[:-1], 't D h w -> (t h w) D')
        one_step_priors = None
        extrapolating_prior = einops.rearrange(log_priors[-1], 'D h w -> (h w) D')
        states = einops.rearrange(states, 't embed_dim h w-> 1 t embed_dim (h w)')
        actions = einops.rearrange(actions, 't act_dim -> 1 t act_dim')

        _, states, h_n = self.one_step_prediction(states[:, :-1], actions[:,:-horizon-1], h0=None, log_priors=one_step_priors)
        h_n = h_n[:,-1]

        state = states[:,-1]
        # extrapolate
        predicted_states, _ = self.extrapolate_latent(state, actions[-horizon-1:], h0=h_n, log_prior=extrapolating_prior)
        predicted_states = torch.cat([state[:,None], predicted_states], dim=1)
        predicted_states = einops.rearrange(predicted_states, 'b t embed_dim (h w) -> b t embed_dim h w', h=h, w=h)[0]
        print(f'predicted_states.shape = {predicted_states.shape}')
        return predicted_states

    def training_step(self, batch, batch_idx):
        # perform predictions and compute loss
        loss, loss_list, entropy_list = self._step(batch)
        #loss_z, loss_v = self._step(batch)
        #loss = loss_z #+ loss_v # TODO re-enable v-loss backprop
        # score and log predictions
        self.log('Training/loss', loss, on_step=True)
        
        figure = plt.figure()
        plt.plot(np.arange(1,len(loss_list)+1,1), torch.tensor(loss_list).detach().cpu().numpy())
        plt.xlabel('Frame')
        plt.ylabel('Loss')
        self.trainer.logger.experiment.add_figure('Training/loss_per_frame', figure, self.global_step)
        #self.log('Training/loss_z', loss_z, on_step=True)
        #self.log('Training/loss_v', loss_v, on_step=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        # perform predictions and compute loss
        loss, loss_list, entropy_list = self._step(batch)
        #loss_z, loss_v = self._step(batch)
        #loss = loss_z #+ loss_v # TODO re-enable v-loss backprop
        # score and log predictions
        self.log('Validation/loss', loss, on_epoch=True)
    
        figure = plt.figure()
        plt.plot(np.arange(1,len(loss_list)+1,1), torch.tensor(loss_list).detach().cpu().numpy())
        plt.xlabel('Frame')
        plt.ylabel('Loss')
        self.trainer.logger.experiment.add_figure('Validation/loss_per_frame', figure, self.global_step)
        
        #self.log('Validation/loss_z', loss_z, on_epoch=True)
        #self.log('Validation/loss_v', loss_v, on_epoch=True)
        return loss
    
    def validation_epoch_end(self, batch_losses):
        # check whether to go to next step in curriculum
        mean_loss = torch.tensor(batch_losses).mean()
        self._check_curriculum_cond(mean_loss)
    
    def configure_optimizers(self):
        # set up optimizer
        params = list(self.gru.parameters()) + list(self.mdn_network.parameters())
        if self.conv_net is not None:
            params = params + list(self.conv_net.parameters())
        optimizer = torch.optim.AdamW(params, **self.hparams.optim_kwargs)
        # set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.hparams.scheduler_kwargs['lr_step_mode'],
            'frequency': self.hparams.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}
    
    def _init_curriculum(self, seq_len=None, curriculum_start=0):
        if seq_len is None:
            seq_len = self.hparams.seq_len
        self.curriculum = [i for i in range(seq_len-2)]
        self.curriculum_step = curriculum_start

    def _check_curriculum_cond(self, value):
        if self.curriculum_step < len(self.curriculum)-1:
            if value < self.hparams.curriculum_threshold:
                self.curriculum_step += 1
                print(f'\nCurriculum updated! New forecast horizon is {self.curriculum[self.curriculum_step]}\n')
        

class RSSM(pl.LightningModule):
    def __init__(self, lstm_kwargs, optim_kwargs, scheduler_kwargs, seq_len, use_pretrained=True, VAE_path=None, VAE_class='Conv'):
        '''
        Adapted from https://arxiv.org/pdf/1811.04551.pdf
        '''
        
        super().__init__()
        
        # save params
        self.save_hyperparameters()

        if use_pretrained:
            # load VAE
            if VAE_path == None:
                raise ValueError('Need to specify VAE path ')
            self.VAE = vae_model_by_str[VAE_class].load_from_checkpoint(VAE_path)
            self.VAE.eval()
            self.latent_dim = self.VAE.hparams.encoder_kwargs['latent_dim']
        else:
            raise NotImplementedError()
            '''
            # init new VAE
            if VAE_kwargs == None:
                raise ValueError('Need to specify VAE kwargs ')
            self.VAE = vae_model_by_str[VAE_class](**VAE_kwargs)
            self.latent_dim = VAE_kwargs['latent_dim']
            '''
        # save some vars
        self.scheduler_kwargs = scheduler_kwargs
        self.optim_kwargs = optim_kwargs
        self.seq_len = seq_len

        # set up model
        self.mse_loss = nn.MSELoss(reduction='none')
        self.merge = util_models.MergeFramesWithBatch()
        self.split = util_models.SplitFramesFromBatch(self.seq_len)
        self.split_cut = util_models.SplitFramesFromBatch(self.seq_len-1)
        lstm_input_dim = self.latent_dim + 128 # s_t-1, a_t-1,  where s_t = [z_t, v_t]
        self.lstm = nn.LSTM(**lstm_kwargs, input_size=lstm_input_dim, batch_first=True)
        self.mdn_network = nn.Sequential(nn.Linear(lstm_kwargs['hidden_size'], 200), nn.ReLU(), nn.Linear(200, (2 * self.latent_dim + 64)))
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.reward_network = nn.Sequential(nn.Linear(2 * (self.latent_dim + 64) + self.latent_dim, 1024), nn.ReLU(), nn.Linear(1024, 1), nn.Sigmoid())
    
    def forward_latent(self, states, actions, h0=None, c0=None, batched=False):
        '''
        Helper function which takes (a sample of the current belief over the) current state or a sequence thereof
        as well as the action taken in that state or states, as well as the current lstm state and computes a belief
        over the next state as well as a prediction of the reward
        Input:
            states - ([B], T, 64 + latent_dim)
            actions - ([B], T, 64)
            h0 - ([B], lstm_kwargs['hidden_size'],)
            c0 - ([B], lstm_kwargs['hidden_size'],)
            batched - Bool, whether pov, vec, actions have a batch dimension before the time dimension
        Output:
            (s_mean, s_std) - belief over state, shape ([B], T, latent_dim + action_dim)
            s_t -  sample from the above factorized normal distribution
            r_t - predicted reward
            (h_n, c_n) - last hidden and cell state of the lstm
        '''
        # concat states and action
        if batched:
            lstm_input = torch.cat([states, actions], dim=2)
        else:
            lstm_input = torch.cat([states, actions], dim=1)[None,...]
        
        # compute hidden states of lstm
        if h0 is None or c0 is None:
            h_t, (h_n, c_n) = self.lstm(lstm_input)
        else:
            h_t, (h_n, c_n) = self.lstm(lstm_input, (h0,c0))
        
        # merge h_t
        h_t = self.merge(h_t) 

        # compute next deterministic state
        s_dist = self.mdn_network(h_t) 
        z_mean, z_logstd = torch.chunk(s_dist[...,:2*self.latent_dim], chunks=2, dim=-1)
        v_mean = s_dist[...,-64:] 
        s_mean = torch.cat([z_mean, v_mean], dim=-1)

        # skip connection for the mean to bias it towards no change
        if batched and len(states.shape) == 3:
            s_mean = s_mean + self.merge(states)
        elif not batched and len(states.shape) == 2:
            s_mean = s_mean + states
        else:
            raise ValueError(f'Unexpected error: batched = {batched} but len(states.shape) = {len(states.shape)} ({states.shape}) ')
        
        #print(f'mean z_logstd = {self.split(s_logstd)[:,:-1,:self.latent_dim].mean()}')
        z_std = torch.exp(z_logstd) # make sure std is non-negative #TODO: could add minimum std here

        # sample from the multi-dim gaussian parameterized by h_t
        s_t = s_mean
        s_t[...,:-64] = s_t[...,:-64] + z_std * torch.normal(torch.zeros_like(z_std), torch.ones_like(z_std))
        
        # predict reward given h_t and s_t
        rew_input = torch.cat([s_mean, z_std, s_t], dim=1)
        r_t = self.reward_network(rew_input)

        return (s_mean, z_std), s_t, r_t, (h_n, c_n)

    def forward(self, pov, vec, actions, h0=None, c0=None, batched=False):
        '''
        Given the last state, latest obs and taken action, this function computes 
        the belief over the next state, as well as predicts the reward.
        Inputs:
            pov - ([B], T, 3, 64, 64)
            vec - ([B], T, 64)
            actions - ([B], T, 64)
            h0 - ([B], lstm_kwargs['hidden_size'],)
            c0 - ([B], lstm_kwargs['hidden_size'],)
            batched - Bool, whether pov, vec, actions have a batch dimension before the time dimension
        Output:
            (s_mean, s_std) - belief over state, shape ([B], T, latent_dim + action_dim)
            s_t -  sample from the above factorized normal distribution
            r_t - predicted reward
            (h_n, c_n) - last hidden and cell state of the lstm
            pov_mean - ([B], T, latent_dim) ground truth state mean
            pov_std - ([B], T, latent_dim) ground truth state std
        '''
        if batched:
            # merge frames with batch
            pov = self.merge(pov)

        # encode pov to latent
        pov_mean, pov_std, pov_sample = self.VAE.encode_only(pov) 
        
        if batched:
            # split frames from batch again
            pov_mean, pov_std, pov_sample = self.split(pov_mean), self.split(pov_std), self.split(pov_sample)
        
        # construct state sample
        states = torch.cat([pov_sample, vec], dim=2 if batched else 1)

        (s_mean, z_std), s_t, r_t, (h_n, c_n) = self.forward_latent(states, actions, h0, c0, batched)        
        
        return (s_mean, z_std), s_t, r_t, (h_n, c_n), pov_mean, pov_std
        

    def _get_log_p(self, x, mean, std):
        '''
        Computes log prob of a x under a diagonal multivariate gaussian
        Shapes:
        x - (B*T, D)
        mu - (B*T, D)
        std - (B*T, D)
        '''
        D = x.shape[1]
        return -0.5 * D * np.log(2*np.pi) - torch.sum(torch.log(std) + (x - mean).abs().pow(2) / (2 * std.abs().pow(2)), dim=1)

    def _step(self, batch):
        '''
        Helper function which encodes the pov obs, cats them with vec obs and action to pass through self.forward
        returns prediction and target
        '''
        # get data
        pov, vec, actions, rew = batch

        # merge frames with batch for batch processing
        merged_vec = self.merge(vec[:,1:,:])
        merged_rew = self.merge(rew[:,1:])

        (s_mean, z_std), s_t, r_t, (h_n, c_n), pov_mean, pov_std = self(pov, vec, actions, batched=True)

        # extract distributions from the tensors
        predicted_z_mean = s_mean[:,:self.latent_dim]
        predicted_z_std = z_std
        #print(f'predicted_z_mean.shape = {predicted_z_mean.shape}')

        predicted_v_mean = s_mean[:,self.latent_dim:]
        #print(f'predicted_v_mean.shape = {predicted_v_mean.shape}')

        # compute log_prob of v_t under its dist
        # cut off last prediction since it can't be scored
        # also cut off first target since it was not predicted
        predicted_v_mean = self.merge(self.split(predicted_v_mean)[:,:-1,:])
        v_loss = self.mse_loss(merged_vec, predicted_v_mean)

        # compute mse of reward (is same as logp under scalar gaussian with unit variance --> see their paper)
        mse_r = self.mse_loss(self.merge(self.split(r_t)[:,:-1,:]).squeeze(), merged_rew)
        
        # compute KL divergence between h_t = (m1, s1) and (pov_mean, pov_std)
        pov_mean, pov_std = self.merge(pov_mean[:,1:,:]), self.merge(pov_std[:,1:,:])
        predicted_z_mean, predicted_z_std = self.merge(self.split(predicted_z_mean)[:,:-1,:]), self.merge(self.split(predicted_z_std)[:,:-1,:])

        # compute KL(enc(o) || pred(z)) in paper, but that seems to lead to bad behavior for us.
        # so for now we comput KL(pred(z) || enc(o))
        # specifically, the predicted std is ~1 oom too large in the KL(enc|pred) case, resulting in
        # very wild extrapolations
        #kld = self._compute_kl((predicted_z_mean, predicted_z_std), (pov_mean, pov_std))
        # Since we are currently training the modules seperately, the pov_mean is not trainable
        # so that the gradient of the KL is the same as the gradient of the following negative log-likelihood:
        # TODO use pov_sample instead of pov_mean
        z_loss = 0.5 * ((predicted_z_mean - pov_mean) / predicted_z_std) ** 2 + torch.log(predicted_z_std)
        z_loss = z_loss.sum(dim=1) 
        #print(f'mean true z std = {pov_std.mean()}')
        #print(f'mean predicted z std = {predicted_z_std.mean()}')
        #print(f'mse std = {self.split_cut((pov_std-predicted_z_std)**2).sum(dim=1).mean()}')
        
        # sum up all losses, split them into frames, sum over frames and average over batch
        v_loss = self.split_cut(v_loss).sum(dim=2).mean() #sum over 2 in deterministic case, since we didn't reduce over the feature dim
        z_loss = self.split_cut(z_loss).mean()
        #print(f'kld = {z_loss}')
        r_loss = self.split_cut(mse_r).mean()
        #print(f'z_loss = {z_loss}')
        #print(f'v_loss = {v_loss}')
        #print(f'r_loss = {r_loss}')
        
        #print(f'pov_std = {pov_std}')
        #print(f'predicted_z_std = {predicted_z_std}')
        #print(f'predicted_v_std = {predicted_v_std}')
        
        return v_loss, z_loss, r_loss
    
    def _compute_kl(self, p, q):
        '''
        Computes KL divergence KL(p || q) between two gaussians p and q with diagonal covariance matrix
        Args:
            p - (mean1, std1), where mean1 and std1 are of shape (B*T, D) with batch dimension B and num frames T
            q - (mean2, std2)
        Returns:
            kld - KL divergence, shape (B*T,)
        '''
        mean1, std1 = p
        mean2, std2 = q
        #print(f'Mean 1 = {mean1.mean()}')
        #print(f'Mean 2 = {mean2.mean()}')
        #print(f'Std 1 = {std1.mean()}')
        #print(f'Std 2 = {std2.mean()}')
        kld = torch.log(std2 / std1) + 0.5 * (std1 ** 2 + (mean2 - mean1) ** 2) / (std2 ** 2) - 0.5#, constant summands don't matter for gradients.
        kld = kld.sum(dim=1)
        #print(f'kld ={kld.mean()}')
        return kld
        
    def training_step(self, batch, batch_idx):
        # perform predictions and compute loss
        v_loss, z_loss, r_loss = self._step(batch)

        # average losses
        loss = (v_loss + z_loss + r_loss) / 3

        # score and log predictions
        self.log('Training/loss', loss, on_step=True)
        self.log('Training/v_loss', v_loss, on_step=True)
        self.log('Training/r_loss', r_loss, on_step=True)
        self.log('Training/z_loss', z_loss, on_step=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        # perform predictions and compute loss
        v_loss, z_loss, r_loss = self._step(batch)
        
        # average losses
        loss = (v_loss + z_loss + r_loss) / 3
        
        # score and log predictions
        self.log('Validation/loss', loss, on_epoch=True)
        self.log('Validation/v_loss', v_loss, on_epoch=True)
        self.log('Validation/r_loss', r_loss, on_epoch=True)
        self.log('Validation/z_loss', z_loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        # set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}

    @torch.no_grad()
    def predict_recursively(self, states, actions, horizon):
        '''
        Auto-regressively applies dynamics model. Actions for imagination are supplied, so only states are being predicted
        Input:
            states - (T, D), where D is latent_dim + obf_vector_dim
            actions - (T + H, D_a), where D_a is obf_action_dim and H is the horizon
            horizon - int, number of time steps to extrapolate
        Output:
            predicted_states - (H, D)
        '''
        assert horizon > 0, f"horizon must be greater 0, but is {horizon}!"

        (s_mean, z_std), s_t, _, (h_n, c_n) = self.forward_latent(states, actions[:-horizon], h0=None, c0=None, batched=False)

        state_list = []
        for t in range(horizon):
            # get last state and action
            s_t = s_t[-1][None,:]
            action = actions[-horizon+t][None,:]
            
            # save state
            state_list.append(s_t)        

            # sample next state
            (s_mean, z_std), s_t, _, (h_n, c_n) = self.forward_latent(s_t, action, h0=h_n, c0=h_n, batched=False)

        # concat states
        predicted_states = torch.cat(state_list, dim=0)

        return predicted_states




class NODEDynamicsModel(pl.LightningModule):
    def __init__(self, base_model_class, base_model_kwargs, VAE_path, optim_kwargs, scheduler_kwargs, seq_len):
        super().__init__()
        
        # save params
        self.save_hyperparameters()
    
        # load VAE
        self.VAE = visual_models.ConvVAE.load_from_checkpoint(VAE_path)
        self.VAE.eval()

        # save some vars
        self.scheduler_kwargs = scheduler_kwargs
        self.optim_kwargs = optim_kwargs
        self.seq_len = seq_len
        self.base_model = base_model_class(**base_model_kwargs)
        self.criterion = nn.MSELoss()
        self.timesteps = None
        self.merge = util_models.MergeFramesWithBatch()
        self.split = util_models.SplitFramesFromBatch(self.seq_len)
        
    
    def forward(self, model_input):
        if self.timesteps is None:
            self.timesteps = torch.linspace(0,self.seq_len,self.seq_len, device=self.device)
        # pass through ode solver
        pred_y = teq.odeint_adjoint(self.base_model, model_input, self.timesteps, adjoint_options={"norm": "seminorm"})
        return pred_y

    def _step(self, batch):
        '''
        Helper function
        '''
        # get data
        pov, vec, actions = batch
        pov = self.merge(pov) # merge frames with batch for batch processing
        pov = self.VAE.encode_only(pov)
        pov = self.split(pov) # split frames from batch again
        obs = torch.cat([pov, vec], dim=2)
        input_obs, target_obs = obs[:,0,:], obs[:,1:,:] # split into input and target
        model_input = torch.cat([input_obs, actions[:,0,:]], dim=1)
        # create predictions
        pred_obs = self(model_input)[:,:,:obs.shape[2]] # throw away the predicted trajectories of actions
        pred_obs = pred_obs[1:,:,:].transpose(0,1) # flip to batch first, and throw away initial value, since it didn't change
        return pred_obs, target_obs
    
    def training_step(self, batch, batch_idx):
        pred_obs, target_obs = self._step(batch)
        # score and log predictions
        loss = self.criterion(pred_obs, target_obs)
        self.log('Training/loss', loss, on_step=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        pred_obs, target_obs = self._step(batch)
        # score and log predictions
        loss = self.criterion(pred_obs, target_obs)
        self.log('Validation/loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        # set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}

class DynamicsBaseModel(nn.Module):
    '''
    Base model for NODEDynamicsModel
    '''
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        hidden_dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], input_dim))

        self.net = nn.Sequential(*layers)
        
    def forward(self, t, model_input):
        '''
        t - time, needed for odeint, but not used in model
        input should be of shape (B, latent_dim + vec_obs_dim + action_dim), e.g. (B, 256)
        '''
        return self.net(model_input)


class DynamicsModel(pl.LightningModule):

    def __init__(self, input_size, num_layers, num_hidden, optim_kwargs, scheduler_kwargs):
        self.save_hyperparameters()

        self.optim_kwargs = optim_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_hidden, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(self.input_size, self.input_size-64) # want to predict latent + vec_obs, not action
        self.criterion = nn.MSELoss()

    def forward(self, input):
        '''
        input should be of shape (B, T, D), where D = L + 64 + 64 and L is the latent dimension of the encoding.
        '''
        print('LSTM input shape', input.shape)
        lstm_out = self.lstm(input)[0] # return last hidden state at every step
        pred = self.linear(lstm_out)
        return pred

    
    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        # set up 
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        print('pred shape', pred.shape)
        # pred should be of same shape as input, i.e. (B, L + 128)
        # pred is scored against original sequence
        loss = self.criterion(pred[:,:-1], batch[:,1:,:-64])
        self.log('Training/loss', loss.mean().item(), on_step=True)
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.criterion(pred[:,:-1], batch[:,1:,:-64])
        self.log('Validation/loss', loss.mean().item(), on_step=True)







class BCLinear(pl.LightningModule):

    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate, scheduler_kwargs, centroids_path, VAE_path):
        super().__init__()
        self.save_hyperparameters()
        
        self.VAE = visual_models.ConvVAE.load_from_checkpoint(VAE_path)
        self.VAE.eval()
        self.centroids = torch.from_numpy(np.load(centroids_path))
        self.learning_rate = learning_rate
        self.scheduler_kwargs = scheduler_kwargs
        self.loss_fct = nn.CrossEntropyLoss()

        hidden_dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.net = nn.Sequential(*layers)
        
    def forward(self, model_input):
        '''
        input should be of shape (B, latent_dim + vec_obs_dim), e.g. (B, 192)
        '''
        return self.net(model_input)

    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # set up 
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}
    
    def training_step(self, batch, batch_idx):
        # get model input and target actions
        pov, vec, actions = batch
        model_input = torch.cat([self.VAE.encode_only(pov), vec], dim=1)
        
        # generate predictions
        pred = self(model_input)
        
        # map action to centroids
        actions = self.remap_actions(actions)
        
        # compute loss and log
        loss = self.loss_fct(pred, actions) 
        self.log('Training/loss', loss.mean().item(), on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # get model input and target actions
        pov, vec, actions = batch
        model_input = torch.cat([self.VAE.encode_only(pov), vec], dim=1)
        
        # generate predictions
        pred = self(model_input)
        
        # map action to centroids
        actions = self.remap_actions(actions)
        
        # compute loss and log
        loss = self.loss_fct(pred, actions) 
        self.log('Validation/loss', loss.mean().item())
        return loss

    @torch.no_grad()
    def remap_actions(self, actions):
        if self.device != self.centroids.device:
            self.centroids = self.centroids.to(self.device)
        # compute distances between action vectors and centroids
        distances = torch.sum((actions - self.centroids[:, None]) ** 2, dim=2)
        # Get the index of the closest centroid to each action.
        # This is an array of (batch_size,)
        actions = torch.argmin(distances, dim=0)
        return actions
    



