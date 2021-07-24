import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torchdiffeq as teq
import utils

import visual_models
import util_models

class PlaNetExperiment():

    def __init__(self, planet_kwargs, env_name):
        self.planet = PlaNet(**planet_kwargs)
        self.env_name = env_name
    
    def run(self):

        # init env
        env = gym.make(self.env_name)
        obs = env.reset()
        rew = 0
        done = False

        
class PlaNet(nn.Module):
    '''
    Adapted from https://arxiv.org/pdf/1811.04551.pdf
    '''
    def __init__(self, rssm_path, action_repeat, max_opt_iter, num_act_sequences, planning_horizon, top_k):
        super()__init__()
        self.rssm = RSSM.load_from_checkpoint(rssm_path)
        self.action_repeat = action_repeat
        self.action_dim = 64
        self.max_opt_iter = max_opt_iter
        self.num_act_sequences = num_act_sequences
        self.planning_horizon = planning_horizon
        self.top_k = top_k

    def forward(self, input):
        pass
    
    def _get_batched_action_sequences(self, dist):
        act = dist.sample(self.num_act_sequence)
        return act.reshape(act.shape[0], self.planning_horizon, self.action_dim)

    def plan(self, s_mean, s_std):
        '''
        s_mean - mean of belief over state, shape (D,)
        s_std - std of belief over state, shape (D,)
        '''
        dist_dim = self.action_dim * self.planning_horizon
        dist = torch.distributions.normal.Normal(loc=torch.zeros(dist_dim), loc=torch.ones(dist_dim))
        a = dist.sample(sample_shape=torch.Size([self.num_act_sequence]))
        print(f'sample shape = {a.shape}')

        for opt_iter in range(self.max_opt_iter):
            # sample action sequence batch
            act_seq = self._get_batched_action_sequences(dist)
            # compute beliefs over next states

            # sample state trajectory from belief

            # predict cumulative reward from state belief and sample

            # pick the K best action sequences

            # re-compute mean and std of dist from the K best action sequences

            # re-parameterize the dist
        
        # return first action mean of the latest distribution

class RSSM(pl.LightningModule):
    def __init__(self, lstm_kwargs, VAE_path, optim_kwargs, scheduler_kwargs, seq_len):
        '''
        Adapted from https://arxiv.org/pdf/1811.04551.pdf
        '''
        
        super().__init__()
        
        # save params
        self.save_hyperparameters()
    
        # load VAE
        self.VAE = visual_models.ConvVAE.load_from_checkpoint(VAE_path)
        self.VAE.eval()
        self.latent_dim = self.VAE.hparams.encoder_kwargs['latent_dim']

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
        self.mdn_network = nn.Sequential(nn.Linear(lstm_kwargs['hidden_size'], 2 * (self.latent_dim + 64)))
        self.elu = nn.ELU()
        self.reward_network = nn.Sequential(nn.Linear(3 * (self.latent_dim + 64), 1), nn.GELU())

    def forward(self, model_input):
        h_t, _ = self.lstm(model_input)
        h_t = self.merge(h_t) # merge frames with batch
        #print(f'h_t.shape = {h_t.shape}')
        s_dist = self.mdn_network(h_t) # compute next deterministic state
        s_mean, s_preact_std = torch.chunk(s_dist, chunks=2, dim=1)
        s_std = self.elu(s_preact_std) + 1 # make sure std is non-negative #TODO: could add minimum std here

        # sample from the multi-dim gaussian parameterized by h_t
        s_t = s_mean + s_std * torch.normal(torch.zeros_like(s_mean), torch.ones_like(s_std))
        
        # predict reward given h_t and s_t
        rew_input = torch.cat([s_mean, s_std, s_t], dim=1)
        #print(f'rew_input.shape = {rew_input.shape}')
        r_t = self.reward_network(rew_input)
        #print(f'r_t.shape = {r_t.shape}')

        return (s_mean, s_std), s_t, r_t

    def _get_log_p(self, x, mean, std):
        '''
        Computes log prob of a x under a diagonal multivariate gaussian
        Shapes:
        x - (B*T, D)
        mu - (B*T, D)
        std - (B*T, D)
        '''
        D = x.shape[1]
        return -0.5 * D * np.log(2*np.pi) - torch.sum(2 * torch.log(std) + (x - mean).abs().pow(2) / (2 * std.abs().pow(2)), dim=1)

    def _step(self, batch):
        '''
        Helper function which encodes the pov obs, cats them with vec obs and action to pass through self.forward
        returns prediction and target
        '''
        # get data
        pov, vec, actions, rew = batch
        
        # merge frames with batch for batch processing
        pov = self.merge(pov)
        merged_vec = self.merge(vec[:,1:,:])
        merged_rew = self.merge(rew[:,1:])
        
        # encode pov to latent
        pov_mean, pov_std, pov_sample = self.VAE.encode_only(pov) 
        
        # split frames from batch again
        pov_sample = self.split(pov_sample)
        
        # prepare model input
        #print(f'pov_sample.shape: {pov_sample.shape}')
        #print(f'vec.shape: {vec.shape}')
        #print(f'actions.shape: {actions.shape}')
        model_input = torch.cat([pov_sample, vec, actions], dim=2)
        
        # create predictions
        (s_mean, s_std), s_t, r_t = self(model_input)

        # extract distributions from the tensors
        predicted_z_mean = s_mean[:,:self.latent_dim]
        predicted_z_std = s_std[:,:self.latent_dim]
        #print(f'predicted_z_mean.shape = {predicted_z_mean.shape}')

        predicted_v_mean = s_mean[:,self.latent_dim:]
        predicted_v_std = s_std[:,self.latent_dim:]
        #print(f'predicted_v_mean.shape = {predicted_v_mean.shape}')

        # compute log_prob of v_t under its dist
        # cut off last prediction since it can't be scored
        # also cut off first target since it was not predicted
        predicted_v_mean = self.merge(self.split(predicted_v_mean)[:,:-1,:])
        predicted_v_std = self.merge(self.split(predicted_v_std)[:,:-1,:])
        logp_v = self._get_log_p(merged_vec, predicted_v_mean, predicted_v_std)
        
        # compute mse of reward (is same as logp under scalar gaussian with unit variance --> see their paper)
        mse_r = self.mse_loss(self.merge(self.split(r_t)[:,:-1,:]).squeeze(), merged_rew)
        
        # compute KL divergence between h_t = (m1, s1) and (pov_mean, pov_std)
        pov_mean, pov_std = self.merge(self.split(pov_mean)[:,1:,:]), self.merge(self.split(pov_std)[:,1:,:])
        predicted_z_mean, predicted_z_std = self.merge(self.split(predicted_z_mean)[:,:-1,:]), self.merge(self.split(predicted_z_std)[:,:-1,:])
        #print(f'predicted_z_mean.shape = {predicted_z_mean.shape}')
        pred_z_dist = (predicted_z_mean, predicted_z_std)
        kld = self._compute_kl((pov_mean, pov_std), pred_z_dist)
        
        # sum up all losses, split them into frames, sum over frames and average over batch
        v_loss = -self.split_cut(logp_v).sum(dim=1).mean()
        latent_loss = self.split_cut(kld).sum(dim=1).mean()
        r_loss = self.split_cut(mse_r).sum(dim=1).mean()
        #print(f'latent_loss = {latent_loss}')
        #print(f'v_loss = {v_loss}')
        #print(f'r_loss = {r_loss}')
        
        #print(f'pov_std = {pov_std}')
        #print(f'predicted_z_std = {predicted_z_std}')
        #print(f'predicted_v_std = {predicted_v_std}')
        
        return v_loss, latent_loss, r_loss
    
    def _compute_kl(self, p, q):
        '''
        Computes KL divergence between two gaussians p and q with diagonal covariance matrix
        Args:
            p - (mean1, std1), where mean1 and std1 are of shape (B*T, D) with batch dimension B and num frames T
            q - (mean2, std2)
        Returns:
            kld - KL divergence, shape (B*T,)
        '''
        mean1, std1 = p
        mean2, std2 = q

        kld = torch.log(std2 / std1) + (std1 ** 2 + (mean1 - mean2) ** 2) / (2 * std2 ** 2) # - 0.5 , constant summands don't matter for gradients.
        kld = kld.sum(dim=1)
        return kld
        
    def training_step(self, batch, batch_idx):
        # perform predictions and compute loss
        v_loss, latent_loss, r_loss = self._step(batch)
        loss = v_loss + latent_loss + r_loss
        # score and log predictions
        self.log('Training/loss', loss, on_step=True)
        self.log('Training/v_loss', v_loss, on_step=True)
        self.log('Training/r_loss', r_loss, on_step=True)
        self.log('Training/latent_loss', latent_loss, on_step=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        # perform predictions and compute loss
        v_loss, latent_loss, r_loss = self._step(batch)
        loss = v_loss + latent_loss + r_loss
        # score and log predictions
        self.log('Validation/loss', loss, on_epoch=True)
        self.log('Validation/v_loss', v_loss, on_epoch=True)
        self.log('Validation/r_loss', r_loss, on_epoch=True)
        self.log('Validation/latent_loss', latent_loss, on_epoch=True)
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
    def predict_recursively(self, input):
        '''
        Auto-regressively applies dynamics model to extrapolate from image
        Input shape should be (B, D), i.e. have no time component yet
        '''
        #TODO make it so that it can take new actions intead of repeating action
        out = input[:,None,:]
        action = out[:,:,-64:]
        for t in range(self.seq_len):
            # predict new frame / latent space
            lstm_out, _ = self.lstm(out)
            mdn_in = lstm_out[:,-1,:] # only take last output
            mean, preact_std = torch.chunk(self.mdn_network(mdn_in), chunks=2, dim=1)
            std = self.elu(preact_std) + 1 # make sure std is non-negative

            # sample from the multi-dim gaussian parameterized by the mdn outputs --> only used to compare with NeuralODE
            pred_z = mean + std * torch.normal(torch.zeros_like(mean), torch.ones_like(std))
            pred_z = pred_z.reshape((pred_z.shape[0],1,pred_z.shape[1]))
            pred_z = torch.cat([pred_z, action], dim=2) # repeat action, #TODO: make it so that action can be given as input?
            
            out = torch.cat([out, pred_z], dim=1) # add new frame to sequence
        
        return out[:,:,:-128] # return generated sequence, but only z part, i.e. not vec obs and vec act.




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
    



