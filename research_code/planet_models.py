import torch
import torch.nn as nn
import torchvision as tv
import gym
import minerl
import numpy as np
from pyvirtualdisplay import Display

import research_code.dynamics_models as dynamics_models
import research_code.reward_model as reward_model

from gym.wrappers import Monitor

STR_TO_CLASS = {
    'mdn': dynamics_models.MDNRNNReward,
    'rssm': dynamics_models.RSSM
}



class PlaNetExperiment():

    def __init__(self, planet_kwargs, env_name, action_repeat, exploration_noise, record=False, video_dir='./videos', max_steps=None):
        self.planet = PlaNet(**planet_kwargs)
        self.action_repeat = action_repeat
        self.env_name = env_name
        self.exploration_noise = exploration_noise
        self.record = record
        self.video_dir = video_dir
        self.max_steps = max_steps

    def run(self):
        # init env
        display = Display(visible=0, size=(300,400))
        display.start()
        if self.record:
            print(f'Saving video to {self.video_dir}')
            env = Monitor(gym.make(self.env_name), self.video_dir, force=True)
        else:
            env = gym.make(self.env_name)
        obs = env.reset()
        done = False
        action = env.action_space.no_op() # init with null action
        total_rew = 0

        h_n = None
        c_n = None
        
        step = 0

        while not done:
            # init obs and act stacks
            pov_list = []
            vec_list = []
            act_list = []
            
            # repeatedly take last computed action and record results
            for r in range(self.action_repeat):
                
                # stack observations and actions
                pov_list.append(tv.transforms.functional.to_tensor(obs['pov']).float())
                vec_list.append(torch.from_numpy(obs['vector']).float())
                act_list.append(torch.from_numpy(action['vector']).float())
                
                # take action and observe
                obs, rew, done, _ = env.step(action)

                # keep track of reward
                total_rew += rew

                # inc step counter
                step += 1
                print(f'Step {step}, Reward: {rew}')

                # check whether done
                if done or step >= self.max_steps:
                    break

            # check whether done
            if done or step >= self.max_steps:
                break

            # stack observations and actions along time dimension
            pov = torch.stack(pov_list, dim=0)
            vec = torch.stack(vec_list, dim=0)
            actions = torch.stack(act_list, dim=0)
            
            # compute belief over state, i.e. encode via VAE
            all_z_mean, all_z_std, all_z_samples = self.planet.model.VAE.encode_only(pov)
            
            
            # construct belief. Since vec is given, it has zero std
            s_mean = torch.cat([all_z_mean, vec], dim=1)
            s_std = torch.cat([all_z_std, torch.zeros_like(vec)], dim=1)
            s_samples = torch.cat([all_z_samples, vec], dim=1)
            
            if self.planet.model_class == 'rssm':
                (s_mean, s_std), s_t, r_t, (h_n, c_n) = self.planet.model.forward_latent(s_samples, actions, h_n, c_n, batched=False)
            elif self.planet.model_class == 'mdn':
                s_t, r_t, (h_n, c_n) = self.planet.model.forward(s_samples, actions, h_n, c_n, batched=False)
            
            # plan from current state
            action = self.planet(s_mean[-1], s_std[-1], h_n, c_n)

            # apply noise
            action += np.random.normal(loc=0, scale=self.exploration_noise, size=action.shape)
            action = {'vector': action}
        
        print(f'Total reward = {total_rew}')
        return total_rew
        
class PlaNet(nn.Module):
    '''
    Adapted from https://arxiv.org/pdf/1811.04551.pdf
    '''
    def __init__(self, model_path, model_class, max_opt_iter, num_act_sequences, planning_horizon, top_k, use_clusters=False, centroids_path='./', reward_model_path=None):
        super().__init__()
        self.model_class = model_class
        if self.model_class == 'rssm':
            self.model = STR_TO_CLASS[model_class].load_from_checkpoint(model_path)
        elif self.model_class == 'mdn':
            self.model = dynamics_models.MDNRNNReward(model_path, reward_model_path)
        self.action_dim = 64
        self.max_opt_iter = max_opt_iter
        self.num_act_sequences = num_act_sequences
        self.planning_horizon = planning_horizon
        self.top_k = top_k
        self.use_clusters = use_clusters
        if self.use_clusters:
            self.centroids = torch.from_numpy(np.load(centroids_path).astype(np.float32))
            self.num_centroids = len(self.centroids)
            print(f'Using {self.num_centroids} discrete action centroids')
        else:
            self.centroids = None

    def forward(self, s_mean, s_std, h_n, c_n):
        '''
        Takes the last observation and lstm state
        '''
        # compute best next action via CEM planning
        action = self.plan(s_mean, s_std, h_n, c_n).detach().cpu().numpy()

        return action
        
    def _get_batched_action_sequences(self, dist):
        if self.use_clusters:
            act = torch.stack([d.sample(sample_shape=torch.Size([self.num_act_sequences])) for d in dist], dim=1)
        else:
            act = dist.sample(sample_shape=torch.Size([self.num_act_sequences]))

        return act.reshape(self.num_act_sequences, self.planning_horizon, -1)
    
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
        cur_state = init_state
        h_n = torch.cat([h0 for _ in range(self.num_act_sequences)], dim=1)
        c_n = torch.cat([c0 for _ in range(self.num_act_sequences)], dim=1)
        reward_sequence = []
        state_sequence = []
        # roll out
        for t in range(self.planning_horizon):
            #
            cur_action = action_sequence[:,t,:][:,None,:]
            cur_state = cur_state[:,None,:]

            #
            if self.model_class == 'rssm':
                (s_mean, s_std), cur_state, r_t, (h_n, c_n) = self.model.forward_latent(cur_state, cur_action, h_n, c_n, batched=True)
            elif self.model_class == 'mdn':
                cur_state, r_t, (h_n, c_n) = self.model(cur_state, cur_action, h_n, c_n, batched=True)
            
            #
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

        # currently we assume that the distribution over action factorizes over timesteps
        
        # init action dist
        if self.use_clusters:
            act_counters = torch.ones(self.num_centroids, self.planning_horizon) # for keeping probs updated
            action_dist = [torch.distributions.Categorical(probs=act_counters[:,i]) for i in range(self.planning_horizon)]
        else:
            dist_dim = self.action_dim * self.planning_horizon 
            action_dist = torch.distributions.normal.Normal(loc=torch.zeros(dist_dim), scale=torch.ones(dist_dim))

        # sample initial states
        s_dist = torch.distributions.Normal(loc=s_mean, scale=s_std)
        s_t = s_dist.sample(sample_shape=torch.Size([self.num_act_sequences]))

        for opt_iter in range(self.max_opt_iter):
            #print(f'Optimization Iteration = {opt_iter+1}')
            
            # sample action sequence batch
            action_sequence = self._get_batched_action_sequences(action_dist)

            # remap indices to centroids
            if self.use_clusters:
                action_sequence = self.centroids[action_sequence.reshape(-1), :].reshape(*action_sequence.shape[:-1], self.action_dim)

            # sample state trajectories and corresponding rewards
            state_sequence, reward_sequence = self._generate_sequence(s_t, action_sequence, h_n, c_n)

            # predict cumulative reward from state belief and sample. 
            rewards = torch.sum(reward_sequence.squeeze(), dim=1)

            # Sort rewards in descending order
            top_k_rewards, top_idcs = torch.sort(rewards, descending=True)
            print(f'Top k expected reward = {top_k_rewards.mean():.4f} +- {top_k_rewards.std():.4f}')

            # Pick the K action sequences with highest expected reward
            top_k_action_sequences = action_sequence[top_idcs, :, :][:self.top_k]

            #TODO: Weight distribution updates with expected reward?
            if self.use_clusters:
                # increase act counter
                best_clusters = torch.argsort(((top_k_action_sequences.reshape(-1, self.action_dim)[None,...] - self.centroids[:,None,:]) ** 2).sum(dim=-1), dim=0)[0]
                best_clusters = best_clusters.reshape(self.top_k, self.planning_horizon)
                for t in range(self.planning_horizon):
                    for a in range(self.top_k):
                        act_counters[best_clusters[a,t], t] += 1
                action_dist = [torch.distributions.Categorical(probs=act_counters[:,i]) for i in range(self.planning_horizon)]

            else:
                # re-compute mean and std of dist from the K best action sequences
                act_mean = torch.mean(top_k_action_sequences, dim=0)
                act_std = torch.std(top_k_action_sequences, dim=0)

                # re-parameterize the dist
                act_dist = torch.distributions.Normal(loc=act_mean.reshape(-1), scale=act_std.reshape(-1))
        
        
        if self.use_clusters:
            # choose as best action the one which has the highest count at time step 1
            best_action = torch.argsort(act_counters[:,0], descending=True)[0]
            print(f'Best action = {best_action}')
            # map the action mean to the closests centroid
            best_action = self.centroids[best_action]
            
        else:
            # return first action mean of the latest distribution
            best_action = act_mean[0]
            
        
        return best_action