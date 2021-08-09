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

vae_model_by_str = {
    'Conv':visual_models.ConvVAE,
    'ResNet':visual_models.ResnetVAE,
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
    def __init__(self, lstm_kwargs, optim_kwargs, scheduler_kwargs, 
                 seq_len, num_components, VAE_path, temp=1, 
                 VAE_class='Conv', skip_connection=True, latent_overshooting=False):
        super().__init__()
        
        # save params
        self.save_hyperparameters()

        # load VAE
        self.VAE = vae_model_by_str[VAE_class].load_from_checkpoint(VAE_path)
        self.VAE.eval()
        if self.hparams.VAE_class == 'vqvae':
            self.latent_dim = self.VAE.hparams.args.embedding_dim
            self.num_embeddings = self.VAE.hparams.args.num_embeddings
            #print(self.VAE.encoder(torch.ones(2,3,64,64).float().to(self.VAE.device)).shape)
            #raise ValueError
            self.latent_size = np.prod(self.VAE.encoder(torch.ones(2,3,64,64).float().to(self.VAE.device)).shape[2:])
            print(f'latent_size (H*W) = {self.latent_size}')
        else:
            self.latent_dim = self.VAE.hparams.encoder_kwargs['latent_dim']

        # set up model
        lstm_input_dim = self.latent_dim * self.latent_size + 64# + 128 # s_t-1, a_t-1,  where s_t = [z_t, v_t]
        self.lstm = nn.LSTM(**lstm_kwargs, input_size=lstm_input_dim, batch_first=True)
        '''self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.VAE.hparams.args.n_hid, out_channels=128, kernel_size=3, padding=1),
            nn.GELU()
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=0),
            nn.Conv2d(),
        )'''
        if self.hparams.VAE_class == 'vqvae':
            self.mdn_network = nn.Sequential(
                                                nn.Linear(lstm_kwargs['hidden_size'], 2024), 
                                                nn.GELU(),
                                                #nn.BatchNorm1d(2024), 
                                                #nn.Linear(2024, 2024),
                                                #nn.GELU(),
                                                #nn.BatchNorm1d(2024),
                                                #nn.Linear(2024, 2024),
                                                #nn.GELU(),
                                                #nn.BatchNorm1d(2024),
                                                nn.Linear(2024, 512),
                                                nn.GELU(),
                                                #nn.BatchNorm1d(2024),
                                                nn.Linear(512, self.num_embeddings * self.latent_size)# + 2 * 64))
                                            ) # first num_component outputs determine the mixing coeffs, the rest parameterize the distributions
        else:
            raise NotImplementedError('Models other than VQVAE are currently not supported')
            self.mdn_network = nn.Sequential(
                                                nn.Linear(lstm_kwargs['hidden_size'], 2024), 
                                                nn.GELU(), 
                                                nn.Linear(2024, 2024),
                                                nn.GELU(),
                                                nn.Linear(2024, self.hparams.num_components + self.hparams.num_components * 2 * (self.latent_dim + 64))
                                            ) # first num_component outputs determine the mixing coeffs, the rest parameterize the gaussians


    def _step(self, batch):
        # unpack batch
        pov, vec, actions, _ = batch

        # predict distribution over next state and sample
        pov_dist_list, _, _, target = self(pov, vec, actions, latent_overshooting=self.hparams.latent_overshooting)
        
        # Compute loss over all time horizons
        loss = 0
        #loss_z = 0
        #loss_v = 0
        T = self.hparams.seq_len-1
        #print(f'len(s_mean_list) = {len(s_mean_list)}')
        loss_list = []
        for t in range(T):
            #print(f't = {t}')
            
            # extract from list
            # For latent overshooting, those are the prior means/logstd/mix_coeffs predicted for state t, 
            # starting from different starting states t' < t.
            pov_dist = pov_dist_list[t] # (B*latent_size, num_starting_states_leading_to_this_one, N_c)
            #print(f'pov_dist.shape = {pov_dist.shape}')
            pov_dist = einops.rearrange(pov_dist, 'b t num_embed latent_size -> b num_embed latent_size t')#, latent_size=self.latent_size)
            #print(f'pov_dist.shape = {pov_dist.shape}')
            # target is the ground truth state at time t, which should be compared to prediction which started from time t
            cur_pov_target = target['pov'][:,t]
            #print(f'cur_target.shape = {cur_target.shape}')

            # get log likelihood of target under distributions, sum over feature dimension
            # target is copied along component and time axis
            # if no latent over shooting then every time step will have only entry
            #logp = -0.5 * (((cur_target[:,None,None,:] - s_means) / torch.exp(s_logstds)) ** 2) - s_logstds
            #logp_z = logp[...,:self.latent_dim].sum(dim=-1)
            #logp_v = logp[...,self.latent_dim:].sum(dim=-1)
            #logp_v = -0.5 * (((cur_vec_target[:,None,None,:] - vec_means) / torch.exp(vec_logstds)) ** 2) - vec_logstds
            #print(f'logp.shape = {logp.shape}')
            
            # compute loss via logsumexp, sum over components, mean over batch and starting states
            #loss += -torch.logsumexp(log_mix_coeffs + logp , dim=1).mean()
            cur_pov_target = einops.repeat(cur_pov_target, 'b latent_size -> b latent_size t', t=t+1)
            #print(f'cur_pov_target.shape = {cur_pov_target.shape}')
            cur_loss = nn.CrossEntropyLoss()(pov_dist, cur_pov_target)
            loss += cur_loss
            loss_list.append(cur_loss)
            #print(f'loss.shape = {loss.shape}')
        
        return loss, loss_list

    def forward(self, pov, vec, actions, h0=None, c0=None, latent_overshooting=False):
        '''
        Given a sequence of pov, vec and actions, computes priors over next latent
        state.
        Inputs:
            pov - ([B], T, 3, 64, 64)
            vec - ([B], T, 64)
            actions - ([B], T, 64)
            h0 - ([B], lstm_kwargs['hidden_size'],)
            c0 - ([B], lstm_kwargs['hidden_size'],)
        Output:
            (s_mean_list, s_logstd_list) - lists of belief over state
                each element is beliefs over the same state but obtained by extrapolating
                from different previous starting states, shape ([B], t, latent_dim + vec_dim), with 1 <= t <= T-1
            s_t -  sample from the above factorized normal distribution #TODO fix so that list is returned
            (h_n, c_n) - last hidden and cell state of the lstm #TODO fix so that list is returned
            log_mix_coeffs_list - list of log of mixing coefficients, ([B] * T, num_components)
            target - ([B], T-1, latent_dim + vec_dim) sample of ground truth encoding
        '''
        
        h_t, c_t = h0, c0

        # merge frames with batch
        b = pov.shape[0]
        seq_len = pov.shape[1]
        pov = einops.rearrange(pov, 'b t c h w -> (b t) c h w')
        
        # encode pov to latent
        if self.hparams.VAE_class == 'vqvae':
            #print(f'pov.shape = {pov.shape}')
            pov_sample, ind = self.VAE.encode_only(pov)
            ind = einops.rearrange(ind, 'bt h w -> bt (h w)')
            #print(f'pov_sample.shape = {pov_sample.shape}')
        else:
            # to do it like in paper, we just use a sample as target
            _, _, pov_sample = self.VAE.encode_only(pov) 

        # split frames from batch again
        pov_sample = einops.rearrange(pov_sample, '(b t) c h w -> b t c (h w)', b=b)
        ind = einops.rearrange(ind, '(b t) hw -> b t hw', b=b)
        #print(f'pov_sample.shape = {pov_sample.shape}')

        states = pov_sample
        #print(f'states.shape = {states.shape}')
        #print(f'ind[:,1:].shape = {ind[:,1:].shape}')
        
        # stack pov and vec to construct target
        if self.hparams.VAE_class == 'vqvae':
            target = {
                'pov': ind[:,1:],
                'vec': vec[:,1:]
            }
        else:
            target = torch.cat([pov_sample, vec], dim=-1)
            target = target[:,1:]


        # latent overshooting
        pov_dist_list = []
        log_mix_coeffs_list = []
        if latent_overshooting:
            # compute one-step predictions
            pov_dist, s_t, (h_t, c_t) = self.forward_latent(states, actions, h_t, c_t, stepwise=True)
            #print(f'pov_dist.shape = {pov_dist.shape}')
            #print(f's_t.shape = {s_t.shape}')
            # save results to list
            pov_dist = einops.rearrange(pov_dist, '(b t hw) c -> b t c hw', b=b, t=seq_len)
            #print(f'pov_dist.shape = {pov_dist.shape}')
            pov_dist_list.extend([[pov_dist[:,i]] for i in range(seq_len-1)])
            #for i in range(seq_len-1):
            #    print(f'pov_dist[:,i].shape = {pov_dist[:,i].shape}')
            #print(f'h_t.shape = {h_t.shape}')
            #print(f'c_t.shape = {c_t.shape}')
            
            # extrapolate/imagine from each state
            for t in range(self.hparams.seq_len-2):
                actions = actions[:,1:]
                starting_state = einops.rearrange(s_t, '(b t) embed_dim latent_size -> b t embed_dim latent_size', b=b)[:,t]
                starting_state = einops.rearrange(starting_state, 'b embed_dim latent_size -> b 1 embed_dim latent_size')
                h_0, c_0 = h_t[:,t,:], c_t[:,t,:]
                _, extrapolated_pov_dist = self.extrapolate_latent(starting_state, actions, h_0, c_0)
                
                # save results to lists
                #print(extrapolated_pov_dist.shape)
                #print(len(pov_dist_list))
                for i in range(extrapolated_pov_dist.shape[1]):
                    #print(f'extrapolated_pov_dist[:,i].shape = {extrapolated_pov_dist[:,i].shape}')
                    pov_dist_list[t+1+i].append(extrapolated_pov_dist[:,i])
                        
        else:
            raise NotImplementedError
            

        # stack lists into tensors
        for i in range(len(pov_dist_list)):
            pov_dist_list[i] = torch.stack(pov_dist_list[i], dim=1)
            
        return pov_dist_list, s_t, (h_t, c_t), target
            
    def forward_latent(self, states, actions, h0=None, c0=None, stepwise=False):
        '''
        Helper function which takes (a sample of the current belief over the) current state or a sequence thereof,
        the action taken in that state or states, as well as the current lstm state and computes a belief
        over the next state
        Input:
            states - ([B], T, 64 + latent_dim)
            actions - ([B], T, 64)
            h0 - ([B], lstm_kwargs['hidden_size'],)
            c0 - ([B], lstm_kwargs['hidden_size'],)
            stepwise - Bool, whether to process input sequentially and store cell state after each time step
        Output:
            (s_mean, s_logstd) - belief over state, shape ([B] * T, num_components * (2 * latent_dim + action_dim))
            s_t -  sample from the above factorized normal distribution
            (h_t, c_t) - hidden and cell state of the lstm after each time step
            log_mix_coeffs - log of mixing coefficients, ([B] * T, num_components)
        '''
        # concat states and action
        #print(f'states.shape = {states.shape}')
        #print(f'actions.shape = {actions.shape}')
        seq_len = actions.shape[1]
        states = einops.rearrange(states, 'b t c hw -> b t (c hw)')
        #actions = einops.repeat(actions, 'b t act_dim -> b t act_dim latent_size', latent_size=states.shape[-1])
        #actions = actions[...,None].repeat(1,1,1,states.shape[-1])
        #print(f'actions.shape = {actions.shape}')

        lstm_input = torch.cat([states, actions], dim=2)
        
        #print(f'lstm_input.shape = {lstm_input.shape}')
        # merge HW with batch
        b = lstm_input.shape[0]
        t = lstm_input.shape[1]
        #lstm_input = einops.rearrange(lstm_input, 'b t D latent_size -> (b latent_size) t D') # D = d_act + d_state
        #lstm_input = lstm_input.permute(0,3,1,2).flatten(start_dim=0, end_dim=1)
        #print(f'lstm_input.shape = {lstm_input.shape}')
        
        
        # compute hidden states of lstm
        # need to do this step wise to have access to c_t at all t. This is really annoying and slow.
        if h0 is None or c0 is None:
            if stepwise:
                h_n_list = []
                c_n_list = []
                h_t, (h_n, c_n) = self.lstm(lstm_input[:,0,:][:,None,:]) # first time step
                h_n_list.append(h_n[0])
                c_n_list.append(c_n[0])
                for i in range(lstm_input.shape[1]-1):
                    h_t, (h_n, c_n) = self.lstm(lstm_input[:,i+1,:][:,None,:], (h_n, c_n)) # time step i+1
                    h_n_list.append(h_n[0])
                    c_n_list.append(c_n[0])
                h_t = torch.stack(h_n_list, dim=1)
                c_t = torch.stack(c_n_list, dim=1)

                ''' Assert passes
                h_t2, (hn2, c_n2) = self.lstm(lstm_input)
                assert (h_t == h_t2).all()
                assert (c_t[:,-1] == c_n2).all()
                '''
            else:
                h_t, (h_n, c_n) = self.lstm(lstm_input)
                c_t = c_n
        else:
            if stepwise:
                h_n_list = []
                c_n_list = []
                h_t, (h_n, c_n) = self.lstm(lstm_input[:,0,:][:,None,:], (h0,c0)) # first time step
                h_n_list.append(h_n[0])
                c_n_list.append(c_n[0])
                for i in range(lstm_input.shape[1]-1):
                    h_t, (h_n, c_n) = self.lstm(lstm_input[:,i+1,:][:,None,:], (h_n, c_n)) # time step i+1
                    h_n_list.append(h_n[0])
                    c_n_list.append(c_n[0])
                h_t = torch.stack(h_n_list, dim=1)
                c_t = torch.stack(c_n_list, dim=1)

                ''' Assert passes
                h_t2, (hn2, c_n2) = self.lstm(lstm_input)
                assert (h_t == h_t2).all()
                assert (c_t[:,-1] == c_n2).all()
                '''
            else:
                h_t, (h_n, c_n) = self.lstm(lstm_input, (h0,c0))
                c_t = c_n
        
        # merge h_t
        #print(f'h_t.shape = {h_t.shape}')
        h_t = einops.rearrange(h_t, 'b t d -> (b t) d', t=seq_len)
        #print(f'h_t.shape = {h_t.shape}')
        # compute next state
        mdn_out = self.mdn_network(h_t) 
        h_t = einops.rearrange(h_t, '(b t) d -> b t d', t=seq_len)
        #print(f'h_t.shape = {h_t.shape}')
        #print(f'mdn_out.shape = {mdn_out.shape}')

        if self.hparams.VAE_class == 'vqvae':
            pov_dist = einops.rearrange(mdn_out, 'bt (num_embeds latent_size) -> (bt latent_size) num_embeds', latent_size=self.latent_size)
        else:
            raise NotImplementedError
            
        # sample from the mixture of multi-dim gaussians parameterized by h_t
        if self.hparams.VAE_class == 'vqvae':
            # sample from discrete dist
            ind = torch.argmax(torch.softmax(pov_dist, dim=-1), dim=-1)
            z_q = self.VAE.quantizer.embed_code(ind)
            #print(f'z_q.shape = {z_q.shape}')
            #print(f'ind.shape = {ind.shape}')
            s_t = einops.rearrange(z_q, '(bt latent_size) embed_dim -> bt embed_dim latent_size', latent_size=self.latent_size)
            return pov_dist, s_t, (h_t, c_t)
        else:
            raise NotImplementedError

    def extrapolate_latent(self, state, actions, h0=None, c0=None):
        '''
        Extrapolate from starting state conditional on actions and pre-existing lstm states
        Args:
            state - ([B], 64 + latent_dim)
            actions - ([B], T', 64), where T' is the number of steps the function will extrapolate
        Returns:
            extrapolated_means - ([B], T', num_components * (64 + latent_dim))
            extrapolated_logstds - ([B], T', num_components * (64 + latent_dim))
        '''
        if h0 is not None:
            h_n = h0[None,:].contiguous()
        else:
            h_n = h0
        if c0 is not None:
            c_n = c0[None,:].contiguous()
        else:
            c_n = c0
        #print(f'h_n.shape = {h_n.shape}')
        #print(f'c_n.shape = {c_n.shape}')

        steps = actions.shape[1] - 1
        b = state.shape[0]
        
        extrapolated_states = []
        extrapolated_pov_dist = []
        
        #print(f'state.shape = {state.shape}')
        #print(f'actions.shape = {actions.shape}')

        for t in range(steps):
            action = actions[:,t][:,None,:]
            #print(state.shape)
            if t == 0:
                state = einops.rearrange(state, 'b t embed_dim latent_size -> b t (embed_dim latent_size)')
            #print(f'state.shape = {state.shape}')
            #print(f'action.shape = {action.shape}')
            
            #action = einops.repeat(action, 'b t d -> (b latent_size) t d', latent_size=self.latent_size)
            #print(f'action.shape = {action.shape}')
            #raise ValueError
            lstm_input = torch.cat([state, action], dim=2)
            #print(f'lstm_input.shape = {lstm_input.shape}')
            #lstm_input = einops.rearrange(lstm_input, 'b t D latent_size -> (b latent_size) t D')
            #print(f'lstm_input.shape = {lstm_input.shape}')
            #print(f'h_n.shape = {h_n.shape}')
            #print(f'c_n.shape = {c_n.shape}')
            h_t, (h_n, c_n) = self.lstm(lstm_input, (h_n, c_n))
            #print(f'h_t.shape = {h_t.shape}')
            h_t = h_t[:,0]
            
            #print(f'h_t.shape = {h_t.shape}')
            
            # compute next state
            mdn_out = self.mdn_network(h_t) 
            pov_dist = einops.rearrange(mdn_out, 'bt (num_embeds latent_size) -> (bt latent_size) num_embeds', latent_size=self.latent_size)

            #print(f'pov_dist.shape = {pov_dist.shape}')
            #print(f'mdn_out.shape = {mdn_out.shape}')

            
            if self.hparams.VAE_class == 'vqvae':
                # sample from discrete dist
                ind = torch.argmax(torch.softmax(pov_dist, dim=-1), dim=-1)
                z_q = self.VAE.quantizer.embed_code(ind)
                #print(f'z_q.shape = {z_q.shape}')
                s_t = einops.rearrange(z_q, '(b latent_size) embed_dim -> b 1 embed_dim latent_size', latent_size=self.latent_size)

            else:
                raise NotImplementedError
            
            pov_dist = einops.rearrange(pov_dist, '(b latent_size) num_embeds -> b num_embeds latent_size', latent_size=self.latent_size)
            #print(f'pov_dist.shape = {pov_dist.shape}')

            extrapolated_states.append(einops.rearrange(state[:,0,:], 'b D -> D b'))
            extrapolated_pov_dist.append(pov_dist)
        
        extrapolated_states = torch.stack(extrapolated_states, dim=1)
        extrapolated_pov_dist = torch.stack(extrapolated_pov_dist, dim=1)
        
        return extrapolated_states, extrapolated_pov_dist

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
        
        seq_len = states.shape[0]
        h = states.shape[2]
        #print(states.shape)
        states = einops.rearrange(states, 't embed_dim h w-> 1 t embed_dim (h w)')
        actions = einops.rearrange(actions, 't act_dim -> 1 t act_dim')
        _, states, (h_n, c_n) = self.forward_latent(states, actions[:,:-horizon], h0=None, c0=None)
        #print(f'h_n.shape = {h_n.shape}')
        #print(f'c_n.shape = {c_n.shape}')
        h_n = h_n[:,-1]
        c_n = c_n[-1]
        print(f'states.shape = {states.shape}')

        state = einops.rearrange(states, 't embed_dim latent_size-> 1 t embed_dim latent_size')[:,-1]
        #print(state.shape)
        state = einops.rearrange(state, 'b embed_dim latent_size -> b 1 embed_dim latent_size')
        #print(state.shape)
        #print(f'h_n.shape = {h_n.shape}')
        #print(f'c_n.shape = {c_n.shape}')
        # extrapolate
        predicted_states, _ = self.extrapolate_latent(state, actions[-horizon:], h0=h_n, c0=c_n)
        print(f'predicted_states[...,0].shape = {predicted_states[...,0].shape}')
        print(f'state[0,0] shape = {state[0,0].shape}')
        state = einops.rearrange(state[0,0], 'embed_dim latent_size -> (embed_dim latent_size) 1')
        predicted_states = torch.cat([state, predicted_states[...,0]], dim=1)
        predicted_states = einops.rearrange(predicted_states, '(embed_dim h w) t -> t embed_dim h w', h=h, w=h)
        #print(f'predicted_states.shape = {predicted_states.shape}')
        return predicted_states

    def training_step(self, batch, batch_idx):
        # perform predictions and compute loss
        loss, loss_list = self._step(batch)
        #loss_z, loss_v = self._step(batch)
        #loss = loss_z #+ loss_v # TODO re-enable v-loss backprop
        # score and log predictions
        self.log('Training/loss', loss, on_step=True)
        self.log('Training/avg_frame_loss', loss/(self.hparams.seq_len-1), on_step=True)
        
        
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
        loss, loss_list = self._step(batch)
        #loss_z, loss_v = self._step(batch)
        #loss = loss_z #+ loss_v # TODO re-enable v-loss backprop
        # score and log predictions
        self.log('Validation/loss', loss, on_epoch=True)
        self.log('Validation/avg_frame_loss', loss/(self.hparams.seq_len-1), on_epoch=True)
        
    
        figure = plt.figure()
        plt.plot(np.arange(1,len(loss_list)+1,1), torch.tensor(loss_list).detach().cpu().numpy())
        plt.xlabel('Frame')
        plt.ylabel('Loss')
        self.trainer.logger.experiment.add_figure('Validation/loss_per_frame', figure, self.global_step)
        
        
        #self.log('Validation/loss_z', loss_z, on_epoch=True)
        #self.log('Validation/loss_v', loss_v, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optim_kwargs)
        # set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.hparams.scheduler_kwargs['lr_step_mode'],
            'frequency': self.hparams.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}

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
    



