import argparse
import os
import torch
import torch.nn as nn

import DQN
import gym
import numpy as np
import einops
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'n_step_state', 'n_step_reward'))

class ReplayMemory(object):

    def __init__(self, capacity, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def add_episode(self, obs, actions, rewards):
        '''
        Adds all transitions within an episode to the memory.
        '''
        discount_array = np.array([self.gamma ** i for i in range(self.n_step)])
        for t in range(len(obs)-self.n_step):
            state = obs[t]
            action = actions[t]
            reward = rewards[t]
            
            if t + self.n_step < len(obs):
                n_step_state = obs[t+self.n_step]
                n_step_reward = np.sum(rewards[t:t+self.n_step] * discount_array)
                next_state = obs[t+1]
            else:
                raise NotImplementedError(f't = {t}, len(obs) = {len(obs)}')
            
            '''elif t + 1 < len(obs):
                n_step_state = None
                n_step_reward = np.sum(rewards[t:] * discount_array[len(rewards[t:])])
                next_state = obs[t+1]
            else:
                n_step_state = None
                n_step_reward = 0
                next_state = None'''
            self.push(
                Transition(
                    state,
                    action,
                    next_state,
                    reward,
                    n_step_state,
                    n_step_reward
                )
            )    
        
def extract_pov_vec(state_list):
    pov = [state['pov'] for state in state_list]
    vec = [state['vector'] for state in state_list]
    return pov, vec

def main(env_name, max_episode_len, model_path, max_env_steps, centroids_path, training_steps_per_iteration,
         lr, n_step, capacity, gamma, action_repeat, epsilon, batch_size):
    
    # set up model
    q_net = DQN.PretrainQNetwork.load_from_checkpoint(model_path)
    
    # set up optimization
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # load centroids
    centroids_path = os.path.join(centroids_path, env_name + '_centroids.npy')
    centroids = np.load(centroids_path)
    
    # init memory
    memory = ReplayMemory(capacity, n_step, gamma)
    
    # log total environment interactions
    total_env_steps = 0
    
    while total_env_steps < max_env_steps:
        
        obs_list = []
        action_list = []
        rew_list = []
        
        env = gym.make(env_name)
        done = False
        obs = env.reset()
        steps = 0
        total_reward = 0
        obs_list.append(obs)
        
        while not done:
            # prepare input
            obs_pov = torch.from_numpy(einops.rearrange(obs['pov'], 'h w c -> 1 c h w').astype(np.float32) / 255).to(q_net.device)
            obs_vec = torch.from_numpy(einops.rearrange(obs['vector'], 'd -> 1 d').astype(np.float32)).to(q_net.device)
            
            # compute q values
            q_values = q_net(obs_pov, obs_vec).squeeze()
            
            # select new action
            if steps % action_repeat == 0:
                if np.random.rand(1)[0] < epsilon:
                    action_ind = np.random.randint(centroids.shape[0])
                else:
                    action_ind = torch.argmax(q_values, dim=0).cpu().item()

                # remap action to centroid
                action = {'vector': centroids[action_ind]}

            # env step
            new_obs, rew, done, _ = env.step(action)
            
            # store transition
            obs_list.append(obs)
            rew_list.append(rew)
            action_list.append(action_ind)
                    
            # bookkeeping
            total_reward += rew
            steps += 1
            total_env_steps += 1
            if steps >= max_episode_len or total_env_steps == max_env_steps:
                break
        print(f'\nTotal reward = {total_reward}')
        
        # store episode into replay memory
        print('\nAdding episode to memory...')
        memory.add_episode(obs_list, action_list, np.array(rew_list))
        
        # perform k updates
        print(f'\nPerforming {training_steps_per_iteration} parameter updates...')
        for _ in range(training_steps_per_iteration):
            # sample batch
            batch = memory.sample(batch_size)
            
            # make batch of Transition objects into Transition object of batches
            batch = Transition(*zip(*batch))
            
            # unpack pov and vec
            pov, vec = extract_pov_vec(batch.state)
            next_pov, next_vec = extract_pov_vec(batch.next_state)
            n_step_pov, n_step_vec = extract_pov_vec(batch.n_step_state)
            
            # prepare tensors
            pov = torch.from_numpy(einops.rearrange(pov, 'b h w c -> b c h w').astype(np.float32)/255).to(q_net.device)
            vec = torch.from_numpy(vec.astype(np.float32)).to(q_net.device)
            next_pov = torch.from_numpy(einops.rearrange(next_pov, 'b h w c -> b c h w').astype(np.float32)/255).to(q_net.device)
            next_vec = torch.from_numpy(next_vec.astype(np.float32)).to(q_net.device)
            n_step_pov = torch.from_numpy(einops.rearrange(n_step_pov, 'b h w c -> b c h w').astype(np.float32)/255).to(q_net.device)
            n_step_vec = torch.from_numpy(n_step_vec.astype(np.float32)).to(q_net.device)
            reward = torch.from_numpy(batch.reward.astype(np.float32)).to(q_net.device)
            n_step_reward = torch.from_numpy(batch.n_step_reward.astype(np.float32)).to(q_net.device)
            
            # compute q values
            q_values = q_net(pov, vec)
            next_q_values = q_net(next_pov, next_vec)
            n_step_q_values = q_net(n_step_pov, n_step_vec)
            
            # compute losses
            one_step_loss = loss_fn(torch.max(q_values, dim=1)[0], reward + gamma * torch.max(next_q_values, dim=1)[0])
            n_step_loss = loss_fn(torch.max(q_values, dim=1)[0], n_step_reward + (gamma ** n_step) * torch.max(n_step_q_values, dim=1)[0])
            loss = one_step_loss + n_step_loss
            total_loss += loss.item()
            
            # backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'\nMean loss = {total_loss / training_steps_per_iteration}')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--centroids_path', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--max_episode_len', type=int, default=4000)
    parser.add_argument('--max_env_steps', type=int, default=2**20)
    parser.add_argument('--n_step', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--action_repeat', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--capacity', type=int, default=20000)
    parser.add_argument('--training_steps_per_iteration', type=int, default=100)
    parser.add_argument('--model_path', help='Path to the (pretrained) DQN')
    
    args = parser.parse_args()
    
    main(**vars(args))