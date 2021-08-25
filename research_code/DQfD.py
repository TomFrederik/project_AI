import argparse
import os
import random
from collections import deque, namedtuple
from time import time

import einops
import gym
import minerl
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import PretrainDQN

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'n_step_state', 'n_step_reward', 'td_error'))

torch.backends.cudnn.benchmark = True

class MemoryDataset(torch.utils.data.Dataset):
    
    def __init__(self, combined_memory):
        '''
        Wrapper class around combined memory to make it compatible with Dataset and be used by DataLoader
        '''
        self.combined_memory = combined_memory
    
    def __len__(self):
        return len(self.combined_memory)
    
    def __getitem__(self, idx):
        state, action, next_state, reward, n_step_state, n_step_reward, td_error = self.combined_memory[idx]
        
        pov = einops.rearrange(state['pov'], 'h w c -> c h w').astype(np.float32) / 255
        next_pov = einops.rearrange(next_state['pov'], 'h w c -> c h w').astype(np.float32) / 255
        n_step_pov = einops.rearrange(n_step_state['pov'], 'h w c -> c h w').astype(np.float32) / 255

        vec = state['vector'].astype(np.float32)
        next_vec = next_state['vector'].astype(np.float32)
        n_step_vec = n_step_state['vector'].astype(np.float32)

        reward = reward.astype(np.float32)
        n_step_reward = n_step_reward.astype(np.float32)
        
        weight = self.weights[idx]

        return (pov, vec), (next_pov, next_vec), (n_step_pov, n_step_vec), action, reward, n_step_reward, idx, weight
    
    def add_episode(self, obs, actions, rewards, td_errors, memory_id):
        self.combined_memory.add_episode(obs, actions, rewards, td_errors, memory_id)
    
    @property
    def weights(self):
        return self.combined_memory.weights

    def update_beta(self, new_beta):
        self.combined_memory.update_beta(new_beta)
    
    def update_td_errors(self, batch_idcs, updated_td_errors):
        self.combined_memory.update_td_errors(batch_idcs, updated_td_errors)
        
class CombinedMemory(object):
    def __init__(self, agent_memory_capacity, n_step, gamma, p_offset, alpha, beta):
        '''
        Class to combine expert and agent memory
        '''
        self.n_step = n_step
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.memory_dict = {
            'expert':ReplayMemory(None, n_step, gamma, p_offset['expert']),
            'agent':ReplayMemory(agent_memory_capacity, n_step, gamma, p_offset['agent'])
        }
    def __len__(self):
        return len(self.memory_dict['expert']) + len(self.memory_dict['agent'])
    
    def add_episode(self, obs, actions, rewards, td_errors, memory_id):
        #time1 = time()
        self.memory_dict[memory_id].add_episode(obs, actions, rewards, td_errors)
        #print(f'Time to add episode = {time() - time1:.2f}s')
    
        # recompute weights
        #time1 = time()
        self._update_weights()
        #print(f'Time to update weights = {time() - time1:.2f}s')
   
    def __getitem__(self, idx):
        return np.concatenate([self.memory_dict['expert'].memory, self.memory_dict['agent'].memory])[idx]

    def sample(self, batch_size):
        idcs = np.random.choice(np.arange(len(self)), size=batch_size, replace=False, p=self.weights)
        #for key in self.memory_dict:
        #    print(key,': ',np.array(self.memory_dict[key].memory).shape)
        return np.concatenate([self.memory_dict['expert'].memory, self.memory_dict['agent'].memory])[idcs], idcs

    def update_beta(self, new_beta):
        for key in self.memory_dict:
            self.memory_dict[key].update_beta(new_beta)
    
    def _update_weights(self):
        weights = np.array([(sars.td_error + self.memory_dict[key].p_offset) ** self.alpha for key in self.memory_dict for sars in self.memory_dict[key].memory])
        #print(weights.shape)
        weights /= np.sum(weights)
        weights = 1 / (len(self) * weights) ** self.beta
        weights /= np.max(weights)
        self.weights = weights / np.sum(weights)
    
    def update_td_errors(self, idcs, td_errors):
        #time1 = time()
        for i, idx in enumerate(idcs):
            if idx < len(self.memory_dict['expert']):
                self.memory_dict['expert'].memory[idx]._replace(td_error=td_errors[i])
            else:
                self.memory_dict['agent'].memory[idx - len(self.memory_dict['expert'])]._replace(td_error=td_errors[i])
        #print(f'Time to update td_errors = {time() - time1:.2f}s')
        
        #time1 = time()        
        self._update_weights()
        #print(f'Time to update weights = {time() - time1:.2f}s')


class ReplayMemory(object):

    def __init__(self, capacity, n_step, gamma, p_offset):
        self.n_step = n_step
        self.gamma = gamma
        self.p_offset = p_offset
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def add_episode(self, obs, actions, rewards, td_errors):
        '''
        Adds all transitions within an episode to the memory.
        '''
        discount_array = np.array([self.gamma ** i for i in range(self.n_step)])
        for t in range(len(obs)-self.n_step):
            state = obs[t]
            action = actions[t]
            reward = rewards[t]
            td_error = td_errors[t]
            
            if t + self.n_step < len(obs):
                n_step_state = obs[t+self.n_step]
                n_step_reward = np.sum(rewards[t:t+self.n_step] * discount_array)
                next_state = obs[t+1]
            else:
                raise NotImplementedError(f't = {t}, len(obs) = {len(obs)}')
            self.push(
                state,
                action,
                next_state,
                reward,
                n_step_state,
                n_step_reward,
                td_error
            )
        
        
    def update_beta(self, new_beta):
        self.beta = new_beta
        
def extract_pov_vec(state_list):
    pov = np.array([state['pov'] for state in state_list])
    vec = np.array([state['vector'] for state in state_list])
    return pov, vec

def load_expert_demo(env_name, data_dir, num_expert_episodes, centroids, combined_memory):
    
    # load data
    print(f"Loading data of {env_name}...")
    data = minerl.data.make(env_name,  data_dir=data_dir)
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)

    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for i, trajectory_name in enumerate(trajectory_names):
        if (i+1) > num_expert_episodes:
            break

        # load trajectory
        print(f'Loading {i+1}th episode...')
        trajectory = list(data.load_data(trajectory_name, skip_interval=0, include_metadata=False))

        # extract lists
        obs = [trajectory[i][0] for i in range(len(trajectory))]
        actions = [np.argmin(np.sum((np.array(trajectory[i][1]['vector'])[None,:] - centroids)**2, axis=1)) for i in range(len(trajectory))]
        rewards = np.array([trajectory[i][2] for i in range(len(trajectory))])
        td_errors = np.ones_like(rewards)

        # add episode to memory
        combined_memory.add_episode(obs, actions, rewards, td_errors, memory_id='expert')
        print(f'Reward: {np.sum(rewards)}\n')


    print('\nLoaded ',len(combined_memory.memory_dict['expert']),' expert samples!')

    return combined_memory

def main(env_name, max_episode_len, model_path, max_env_steps, centroids_path, training_steps_per_iteration,
         lr, n_step, capacity, gamma, action_repeat, epsilon, batch_size, num_expert_episodes, data_dir, save_dir,
         alpha, beta_0, agent_p_offset, expert_p_offset):
    
    # set save dir
    save_dir = os.path.join(save_dir, env_name, str(int(time())))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'q_net.pt')
    print(f'\nSaving model to {save_path}!')
    writer = SummaryWriter(log_dir=save_dir)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # log time
    start = time()

    # set up model
    q_net = PretrainDQN.PretrainQNetwork.load_from_checkpoint(model_path).to(device)
    target_net = PretrainDQN.PretrainQNetwork.load_from_checkpoint(model_path).to(device)
    target_net.eval()
    
    # set up optimization
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')
    
    # load centroids
    centroids_path = os.path.join(centroids_path, env_name + '_centroids.npy')
    centroids = np.load(centroids_path)
    
    # init memory
    beta = beta_0
    combined_memory = CombinedMemory(capacity, n_step, gamma, {'agent':agent_p_offset, 'expert':expert_p_offset}, alpha, beta)
    # init expert memory
    combined_memory = load_expert_demo(env_name, data_dir, num_expert_episodes, centroids, combined_memory)
    
    
    # init the dataset
    dataset = MemoryDataset(combined_memory)
    
    # log total environment interactions
    total_env_steps = 0
    num_episodes = 0

    # create env    
    env = gym.make(env_name)

    time1 = time()
    while total_env_steps < max_env_steps:
        
        obs_list = []
        action_list = []
        rew_list = []
        td_error_list = []
        
        
        num_episodes += 1
        print(f'\nStarting episode {num_episodes}...')

        # re-init env
        done = False
        obs = env.reset()
        steps = 0
        total_reward = 0
        obs_list.append(obs)
        # prepare input
        obs_pov = torch.from_numpy(einops.rearrange(obs['pov'], 'h w c -> 1 c h w').astype(np.float32) / 255).to(q_net.device)
        obs_vec = torch.from_numpy(einops.rearrange(obs['vector'], 'd -> 1 d').astype(np.float32)).to(q_net.device)

        # go to eval mode
        q_net.eval()
        
        with torch.no_grad():
            # compute q values
            q_values = q_net(obs_pov, obs_vec).squeeze()
            time0 = time()        
            while not done:    
                
                # select new action
                #time1 = time()
                if steps % action_repeat == 0:
                    if np.random.rand(1)[0] < epsilon:
                        action_ind = np.random.randint(centroids.shape[0])
                        highest_q = q_values[action_ind].cpu().item()
                    else:
                        action_ind = torch.argmax(q_values, dim=0).cpu().item()
                        highest_q = q_values[action_ind].cpu().item()

                    # remap action to centroid
                    action = {'vector': centroids[action_ind]}
                #print(f'Selecting an action took {time()-time1}s')
                
                # env step
                #time1 = time()
                obs, rew, done, _ = env.step(action)
                
                # store transition
                obs_list.append(obs)
                rew_list.append(rew)
                action_list.append(action_ind)
                
                #print(f'Taking a step and storing transition took {time()-time1}s')
                
                # prepare input
                #time1 = time()
                obs_pov = torch.from_numpy(einops.rearrange(obs['pov'], 'h w c -> 1 c h w').astype(np.float32) / 255).to(q_net.device)
                obs_vec = torch.from_numpy(einops.rearrange(obs['vector'], 'd -> 1 d').astype(np.float32)).to(q_net.device)
                #print(f'Preparing input took {time()-time1}s')
                
                # compute q values
                #time1 = time()
                q_values = q_net(obs_pov, obs_vec).squeeze()
                #print(f'Computing q_values took {time()-time1}s')

                # record td_error
                #time1 = time()
                td_error_list.append(np.abs(rew + gamma * target_net(obs_pov, obs_vec).squeeze()[torch.argmax(q_values)].cpu().item() - highest_q))
                #print(f'Computing td_error took {time()-time1}s')
                        
                # bookkeeping
                total_reward += rew
                steps += 1
                #print(steps)
                total_env_steps += 1
                if steps >= max_episode_len or total_env_steps == max_env_steps:
                    break
            
        print(f'\nEpisode {num_episodes}: Total reward: {total_reward}, Duration: {time()-time0}s')
        
        # store episode into replay memory
        print('\nAdding episode to memory...')
        dataset.add_episode(obs_list, action_list, np.array(rew_list), td_error_list, memory_id='agent')
        #print(dataset[0])
        #print(len(dataset))
    
        # init/update sampler and loader
        sampler = torch.utils.data.WeightedRandomSampler(replacement=True, num_samples=training_steps_per_iteration * batch_size, weights=dataset.weights)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=6)
        
        # perform k updates
        print(f'\nPerforming {training_steps_per_iteration} parameter updates...')
        total_loss = 0
        updated_td_errors = {}
        
        # go to train mode
        q_net.train()

        for batch in tqdm(iter(dataloader)):
            # unpack batch
            #time1 = time()
            state, next_state, n_step_state, action, reward, n_step_reward, batch_idcs, weights = batch
            #print(f'Unpacking batch took {time()-time1}s')
            
            # unpack pov and vec
            pov, vec = state
            next_pov, next_vec = next_state
            n_step_pov, n_step_vec = n_step_state

            # prepare tensors
            #time1 = time()
            pov = pov.to(device)
            vec = vec.to(device)
            next_pov = next_pov.to(device)
            next_vec = next_vec.to(device)
            n_step_pov = n_step_pov.to(device)
            n_step_vec = n_step_vec.to(device)
            reward = reward.to(device)
            n_step_reward = n_step_reward.to(device)
            action = action.to(device)
            weights = weights.to(device)
            #print(f'preparing input took {time()-time1}s')
            
            # compute q values and choose actions
            #time1 = time()
            q_values = q_net(pov, vec)
            #print(f'inferencing q_values took {time()-time1}s')
            #time1 = time()
            next_q_values = target_net(next_pov, next_vec).detach()
            #print(f'inferencing next_q_values took {time()-time1}s')
            #time1 = time()
            base_next_action = torch.argmax(next_q_values, dim=1)
            #print(f'inferencing base_next_action took {time()-time1}s')
            #time1 = time()
            n_step_q_values = target_net(n_step_pov, n_step_vec).detach()
            #print(f'inferencing n_step_q_values took {time()-time1}s')
            #time1 = time()
            base_n_step_action = torch.argmax(n_step_q_values, dim=1)
            #print(f'inferencing base_n_step_action took {time()-time1}s')
            
            # compute losses
            #time1 = time()
            idcs = torch.arange(0, len(q_values), dtype=torch.long, requires_grad=False)
            #print(f'Computing losses took {time()-time1}s')
            #time1 = time()
            selected_q_values = torch.gather(q_values, 1, action[:,None])
            #print(f'Indexing q_values took {time()-time1}s')
            #time1 = time()
            selected_next_q_values = torch.gather(next_q_values, 1, base_next_action[:,None])
            #print(f'indexing next_q_values took {time()-time1}s')
            #time1 = time()
            selected_n_step_q_values = torch.gather(n_step_q_values, 1, base_n_step_action[:,None])
            #print(f'Indexing n_step_q_values took {time()-time1}s')
            #time1 = time()

            one_step_td_errors = reward + gamma * selected_next_q_values - selected_q_values
            #print(f'Computing one_step_td_errors took {time()-time1}s')
            #time1 = time()
            one_step_loss = ((one_step_td_errors ** 2) * weights).mean() # importance sampling scaling
            #print(f'Computing one_step_loss took {time()-time1}s')
            #time1 = time()
            
            n_step_td_errors = reward + (gamma ** n_step) * selected_n_step_q_values - selected_q_values
            #print(f'Computing n_step_td_errors took {time()-time1}s')
            #time1 = time()
            n_step_loss = ((n_step_td_errors ** 2) * weights).mean() # importance sampling scaling
            #print(f'Computing n_step_loss took {time()-time1}s')
            #time1 = time()
            loss = one_step_loss + n_step_loss
            #print(f'Computing losses took {time()-time1}s')
            #time1 = time()
            total_loss += loss
            #print(f'Updating total loss took {time()-time1}s')
            
            # update td errors
            #time1 = time()
            # update towards n_step td error since that ought to be a more accurate estimate of the 'true' error
            dataset.update_td_errors(batch_idcs, torch.abs(n_step_td_errors))
            #print(f'Updating td errors took {time()-time1}s')
            
            # backward pass and update
            time1 = time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(f'Backward+Step took {time()-time1}s')
            
        print(f'\nMean loss = {total_loss.item() / training_steps_per_iteration}')
        cur_dur = time()-start
        print(f'Time elapsed so far: {cur_dur // 60}m {cur_dur % 60:.1f}s')
        print(f'Time per iteration: {cur_dur / num_episodes:.1f}s')
        print('\nUpdating target...')
        target_net.load_state_dict(q_net.state_dict())
        print('\nSaving model')
        torch.save(q_net.state_dict(), save_path)
        print('\nUpdating beta...')
        beta = min(beta + 0.01, 1)
        dataset.update_beta(beta)
        print('\nUpdating Dataloader...') # only update sampling weights after training loop 
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size, 
            sampler=torch.utils.data.WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset), replacement=True),
            num_workers=6
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--centroids_path', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--save_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--max_episode_len', type=int, default=4000)
    parser.add_argument('--max_env_steps', type=int, default=2**20)
    parser.add_argument('--num_expert_episodes', type=int, default=10)
    parser.add_argument('--n_step', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--action_repeat', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.4, help='PER exponent')
    parser.add_argument('--beta_0', type=float, default=0.6, help='Initial PER Importance Sampling exponent')
    parser.add_argument('--agent_p_offset', type=float, default=0.001)
    parser.add_argument('--expert_p_offset', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--capacity', type=int, default=20000)
    parser.add_argument('--training_steps_per_iteration', type=int, default=100)
    parser.add_argument('--model_path', help='Path to the (pretrained) DQN')
    
    args = parser.parse_args()
    
    main(**vars(args))
