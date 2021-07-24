import torch
import torch.nn as nn
import torchvision as tv
import pytorch_lightning as pl
import numpy as np
import utils
import gym

import visual_models
import util_models
import dynamics_models


class PlaNetExperiment():

    def __init__(self, planet_kwargs, env_name, action_repeat):
        self.planet = PlaNet(**planet_kwargs)
        self.action_repeat = action_repeat
        self.env_name = env_name
    
    def run(self):
        # init env
        env = gym.make(self.env_name)
        obs = env.reset()
        done = False
        action = env.action_space.noop()

        total_rew = 0

        h_n = torch.zeros(0)
        
        while not done:
            pov_list = []
            vec_list = []
            act_list = []
            for _ in range(self.action_repeat):
                rew, obs, done, _ = env.step(action)
                
                # stack observations and actions
                pov_list.append(tv.transform.pil_to_tensor(obs['pov']))
                vec_list.append(torch.from_numpy(obs['vector']))
                act_list.append(torch.from_numpy(action))
                
                # keep track of reward
                total_rew += rew
                
                # check whether done
                if done: break

            # check whether done
            if done: break
            
            # stack observations and actions along time dimension
            pov = torch.stack(pov_list, dim=0)
            vec = torch.stack(vec_list, dim=0)
            act = torch.stack(act_list, dim=0)
            
            # plan from current state
            action = self.planet(pov, vec, h_n, c_n)
        
        print(f'Total reward = {total_rew}')
        return total_rew
        
class PlaNet(nn.Module):
    '''
    Adapted from https://arxiv.org/pdf/1811.04551.pdf
    '''
    def __init__(self, rssm_path, max_opt_iter, num_act_sequences, planning_horizon, top_k):
        super().__init__()
        self.rssm = dynamics_models.RSSM.load_from_checkpoint(rssm_path)
        self.action_dim = 64
        self.max_opt_iter = max_opt_iter
        self.num_act_sequences = num_act_sequences
        self.planning_horizon = planning_horizon
        self.top_k = top_k

    def forward(self, pov, vec, h_n, c_n):
        '''
        Takes the last observation and lstm state
        '''
        # compute belief over state, i.e. encode via VAE
        z_mean, z_std, z_sample = self.rssm.VAE.encode_only(pov)
        
        # construct belief. Since vec is given, it has zero std
        s_mean = torch.cat([z_mean, vec], dim=1)
        s_std = torch.cat([z_std, torch.zeros_like(vec)], dim=1)

        # compute best next action via CEM planning
        action = self.plan(s_mean, s_std, h_n, c_n).detach().cpu().numpy()

        return action
        
    def _get_batched_action_sequences(self, dist):
        act = dist.sample(self.num_act_sequence)
        return act.reshape(act.shape[0], self.planning_horizon, self.action_dim)
    
    def _generate_sequence(self, init_state, action_sequence, h0, c0):
        '''
        Given an initial state and lstm state as well as an action sequence, this function
        samples a state sequence from the rssm model and predicts the rewards gained in those states.
        Input:
            init_state - (B, D), where B is self.num_act_sequences
            action_sequence - (B, T, D_act), where T is self.planning_horizon
            h0 - initial lstm hidden state
            c0 - initial lstm cell state
        Output:
            state_sequence - (B, T, D)
            reward_sequence - (B, T)
        '''
        # init
        cur_state = init_state[:,None,:]
        h_n = h0
        c_n = c0
        reward_sequence = []
        state_sequence = []
        
        # roll out
        for t in range(self.planning_horizon):
            cur_action = action_sequence[:,t,:][:,None,:]
            (s_mean, s_std), cur_state, r_t, (h_n, c_n) = self.rssm.forward_latent(cur_state[:,None,:], cur_action, h_n, c_n, batched=True)
            reward_sequence.append(r_t)
            state_sequence.append(cur_state)
        
        # stack along time dimension
        state_sequence = torch.cat(state_sequence, dim=1)
        reward_sequence = torch.stack(reward_sequence, dim=1)
        
        return state_sequence, reward_sequence

    def plan(self, s_mean, s_std, h_n, c_n):
        '''
        s_mean - mean of belief over state, shape (D,)
        s_std - std of belief over state, shape (D,)
        h_n - last hidden state of lstm
        c_n - last cell state of lstm
        '''
        dist_dim = self.action_dim * self.planning_horizon
        action_dist = torch.distributions.normal.Normal(loc=torch.zeros(dist_dim), scale=torch.ones(dist_dim))
        act_sample = action_dist.sample(sample_shape=torch.Size([self.num_act_sequence]))
        print(f'act_sample shape = {act_sample.shape}')

        # sample initial states
        s_dist = torch.distributions.Normal(loc=s_mean, scale=s_std)
        s_t = s_dist.sample(sample_shape=torch.Size([self.num_act_sequences]))
        print(f's_t shape = {s_t.shape}')


        for opt_iter in range(self.max_opt_iter):
            # sample action sequence batch
            action_sequence = self._get_batched_action_sequences(action_dist)

            # sample state trajectories and corresponding rewards
            state_sequence, reward_sequence = self._generate_sequence(s_t, action_sequence, h_n, c_n)

            # predict cumulative reward from state belief and sample
            rewards = torch.sum(reward_sequence, dim=1)

            # pick the K best action sequences
            top_k_action_sequences = action_sequence[torch.argsort(rewards), :, :][:self.top_k]

            # re-compute mean and std of dist from the K best action sequences
            act_mean = torch.mean(top_k_action_sequences, dim=0)
            act_std = torch.std(top_k_action_sequences, dim=0)

            # re-parameterize the dist
            act_dist = torch.distributions.Normal(loc=act_mean.reshape(-1), scale=act_std.reshape(-1))
        
        # return first action mean of the latest distribution
        return act_mean[0]