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

from PretrainDQN import ConvFeatureExtractor, QNetwork

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'n_step_state', 'n_step_reward', 'td_error', 'expert'))


class MemoryDataset(torch.utils.data.Dataset):
    
    def __init__(self, combined_memory):
        '''
        Wrapper class around combined memory to make it compatible with Dataset and be used by DataLoader
        '''
        self.combined_memory = combined_memory
    
    def __len__(self):
        return len(self.combined_memory)
    
    def __getitem__(self, idx):
        state, action, next_state, reward, n_step_state, n_step_reward, td_error, expert = self.combined_memory[idx]
        
        pov = einops.rearrange(state['pov'], 'h w c -> c h w').astype(np.float32) / 255
        next_pov = einops.rearrange(next_state['pov'], 'h w c -> c h w').astype(np.float32) / 255
        n_step_pov = einops.rearrange(n_step_state['pov'], 'h w c -> c h w').astype(np.float32) / 255

        vec = state['vector'].astype(np.float32)
        next_vec = next_state['vector'].astype(np.float32)
        n_step_vec = n_step_state['vector'].astype(np.float32)

        reward = reward.astype(np.float32)
        n_step_reward = n_step_reward.astype(np.float32)
        
        weight = self.weights[idx]

        return (pov, vec), (next_pov, next_vec), (n_step_pov, n_step_vec), action, reward, n_step_reward, idx, weight, expert
    
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
            'expert':ReplayMemory(None, n_step, gamma, p_offset['expert'], expert=True),
            'agent':ReplayMemory(agent_memory_capacity, n_step, gamma, p_offset['agent'], expert=False)
        }
        self.concat_memo = np.concatenate([self.memory_dict['expert'].memory, self.memory_dict['agent'].memory])
    
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
        if memory_id == 'expert': # TODO do this in a less hacky way
            self.concat_memo = self.memory_dict[memory_id].memory

        elif memory_id == 'agent':
            print(len(self.memory_dict['expert'].memory))
            print(len(self.memory_dict['agent'].memory))
            self.concat_memo = np.concatenate([self.memory_dict['expert'].memory, self.memory_dict['agent'].memory])
   
    def __getitem__(self, idx):
        return self.concat_memo[idx]

    def sample(self, batch_size):
        idcs = np.random.choice(np.arange(len(self)), size=batch_size, replace=False, p=self.weights)
        return self.concat_memo[idcs], idcs

    def update_beta(self, new_beta):
        for key in self.memory_dict:
            self.memory_dict[key].update_beta(new_beta)
    
    def _update_weights(self):
        weights = np.array([(sars.td_error + self.memory_dict[key].p_offset) ** self.alpha for key in ['expert', 'agent'] for sars in self.memory_dict[key].memory])
        #print(weights.shape)
        weights /= np.sum(weights) # = P(i)
        weights = 1 / (len(self) * weights) ** self.beta
        self.weights = weights / np.max(weights)
    
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

    def __init__(self, capacity, n_step, gamma, p_offset, expert=False):
        self.n_step = n_step
        self.gamma = gamma
        self.p_offset = p_offset
        self.memory = deque([],maxlen=capacity)
        self.expert = int(expert)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def add_episode(self, obs, actions, rewards, td_errors):
        '''
        Adds all transitions within an episode to the memory.
        '''
        assert len(obs) > self.n_step, f"Expected len(obs) > self.n_step, but are {len(obs)} and {self.n_step}!"
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
                td_error,
                self.expert
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
    
    torch.manual_seed(1337)
    np.random.seed(1337)

    # set save dir
    save_dir = os.path.join(save_dir, 'DQfD', env_name, str(int(time())))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'q_net.pt')
    print(f'\nSaving model to {save_path}!')
    writer = SummaryWriter(log_dir=save_dir)
    

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # log time
    start = time()

    # set up model
    q_net = QNetwork.load_from_checkpoint(model_path).to(device)
    
    # set up optimization
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')
    
    # load centroids
    centroids_path = os.path.join(centroids_path, env_name + '_150_centroids.npy') #TODO make sure that it uses the same centroids as in pretraining
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
        time1 = time()
        obs = env.reset()
        print(f'Resetting the environment took {time()-time1}s')
        
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
            q_values = q_net(obs_pov, obs_vec)[0].squeeze()
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
                q_values = q_net(obs_pov, obs_vec)[0].squeeze()
                #print(f'Computing q_values took {time()-time1}s')

                # record td_error
                #time1 = time()
                td_error_list.append(np.abs(rew + gamma * q_net(obs_pov, obs_vec, target=True)[0].squeeze()[torch.argmax(q_values)].cpu().item() - highest_q))
                #print(f'Computing td_error took {time()-time1}s')
                        
                # bookkeeping
                total_reward += rew
                steps += 1
                #print(steps)
                total_env_steps += 1
                if steps >= max_episode_len or total_env_steps == max_env_steps:
                    break

        print(f'\nEpisode {num_episodes}: Total reward: {total_reward}, Duration: {time()-time0}s')
        writer.add_scalar('Training/EpisodeReward', total_reward, global_step=num_episodes)

        # store episode into replay memory
        print('\nAdding episode to memory...')
        dataset.add_episode(obs_list, action_list, np.array(rew_list), td_error_list, memory_id='agent')
        
        # perform k updates
        print(f'\nPerforming {training_steps_per_iteration} parameter updates...')
        total_loss = 0
        updated_td_errors = {}
        
        # go to train mode
        q_net.train()

        for i in tqdm(range(training_steps_per_iteration)):
            batch_idcs = torch.multinomial(torch.from_numpy(dataset.weights), replacement=False, num_samples=batch_size)

            # unpack batch
            #time1 = time()
            batch = [dataset[idx] for idx in batch_idcs]
            state, next_state, n_step_state, action, reward, n_step_reward, batch_idcs, weights, expert_mask = zip(*batch)
            #print(f'Unpacking batch took {time()-time1}s')

            pov, vec = map(lambda x: np.array(x), zip(*state))
            next_pov, next_vec = map(lambda x: np.array(x), zip(*next_state))
            n_step_pov, n_step_vec = map(lambda x: np.array(x), zip(*n_step_state))

            # prepare tensors
            pov = torch.from_numpy(pov).to(device)
            vec = torch.from_numpy(vec).to(device)
            next_pov = torch.from_numpy(next_pov).to(device)
            next_vec = torch.from_numpy(next_vec).to(device)
            n_step_pov = torch.from_numpy(n_step_pov).to(device)
            n_step_vec = torch.from_numpy(n_step_vec).to(device)
            reward = torch.from_numpy(np.array(reward)).to(device)
            n_step_reward = torch.from_numpy(np.array(n_step_reward)).to(device)
            action = torch.from_numpy(np.array(action)).to(device)
            weights = torch.from_numpy(np.array(weights)).to(device)
            expert_mask = torch.from_numpy(np.array(expert_mask)).to(device)
            
            # compute q values and choose actions
            q_values = q_net(pov, vec)[0]
            next_target_q_values = q_net(next_pov, next_vec, target=True).detach()
            next_q_values = q_net(next_pov, next_vec).detach()
            next_action = torch.argmax(next_q_values, dim=1)
            n_step_q_values = q_net(n_step_pov, n_step_vec, target=True).detach()
            n_step_action = torch.argmax(n_step_q_values, dim=1)
            
            # compute losses
            idcs = torch.arange(0, len(q_values), dtype=torch.long, requires_grad=False)
            selected_q_values = q_values[idcs, action]
            selected_next_q_values = next_target_q_values[idcs, next_action]
            selected_n_step_q_values = n_step_q_values[idcs, n_step_action]

            td_error = reward + gamma * next_q_values[idcs, next_action] - q_values[idcs, action]

            J_DQ = (reward + gamma * selected_next_q_values - selected_q_values)**2
            one_step_loss = (J_DQ * weights).mean() # importance sampling scaling
            
            n_step_td_errors = reward + (gamma ** n_step) * selected_n_step_q_values - selected_q_values
            n_step_loss = ((n_step_td_errors ** 2) * weights).mean() # importance sampling scaling

            loss = one_step_loss + n_step_loss 
            J_E = (expert_mask * q_net._large_margin_classification_loss(q_values, action)).mean()
            loss = loss + J_E
            total_loss += loss
            
            # update td errors
            # update towards n_step td error since that ought to be a more accurate estimate of the 'true' error
            dataset.update_td_errors(batch_idcs, torch.abs(n_step_td_errors))
            
            # backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = total_loss.item() / training_steps_per_iteration
        print(f'\nMean loss = {mean_loss}')
        writer.add_scalar('Training/Loss', mean_loss, global_step=num_episodes)

        cur_dur = time()-start
        print(f'Time elapsed so far: {cur_dur // 60}m {cur_dur % 60:.1f}s')
        print(f'Time per iteration: {cur_dur / num_episodes:.1f}s')
        print('\nUpdating target...')
        q_net._update_target()
        print('\nSaving model')
        torch.save(q_net.state_dict(), save_path)
        print('\nUpdating beta...')
        beta = min(beta + 0.01, 1)
        dataset.update_beta(beta)

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
