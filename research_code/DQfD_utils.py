from collections import deque, namedtuple

import einops
import minerl
import numpy as np
import random
import torch

Transition = namedtuple('Transition',
                        (
                            'state', 
                            'action', 
                            'predictive_state', 
                            'next_state', 
                            'next_predictive_state',
                            'reward', 
                            'n_step_state', 
                            'n_step_predictive_state',
                            'n_step_reward', 
                            'td_error', 
                            'expert'
                        )
                    )

class MemoryDataset(torch.utils.data.Dataset):
    
    def __init__(self, env_name, data_dir, num_expert_episodes, centroids, combined_memory_kwargs, dynamics_model=None):
        '''
        Wrapper class around combined memory to make it compatible with Dataset and be used by DataLoader
        '''
        self.combined_memory = CombinedMemory(**combined_memory_kwargs)
        self.dynamics_model = dynamics_model
        
        # init expert memory
        self._load_expert_demo(env_name, data_dir, num_expert_episodes, centroids, dynamics_model)
    
    def __len__(self):
        return len(self.combined_memory)
    
    def __getitem__(self, idx):
        state, action, predictive_state, next_state, next_predictive_state, reward, n_step_state, n_step_predictive_state, n_step_reward, td_error, expert = self.combined_memory[idx]
        pov = einops.rearrange(state['pov'], 'h w c -> c h w').astype(np.float32) / 255
        next_pov = einops.rearrange(next_state['pov'], 'h w c -> c h w').astype(np.float32) / 255
        n_step_pov = einops.rearrange(n_step_state['pov'], 'h w c -> c h w').astype(np.float32) / 255

        vec = state['vector'].astype(np.float32)
        next_vec = next_state['vector'].astype(np.float32)
        n_step_vec = n_step_state['vector'].astype(np.float32)

        reward = reward.astype(np.float32)
        n_step_reward = n_step_reward.astype(np.float32)
        
        weight = self.weights[idx]

        return (pov, vec), (next_pov, next_vec), (n_step_pov, n_step_vec), predictive_state, next_predictive_state, n_step_predictive_state, action, reward, n_step_reward, idx, weight, expert
    
    def _load_expert_demo(self, env_name, data_dir, num_expert_episodes, centroids, dynamics_model):
        self.combined_memory = load_expert_demo(env_name, data_dir, num_expert_episodes, centroids, self.combined_memory, dynamics_model)

    def add_episode(self, obs, actions, rewards, td_errors, predictive_state, memory_id):
        self.combined_memory.add_episode(obs, actions, rewards, td_errors, predictive_state, memory_id)
    
    @property
    def weights(self):
        return self.combined_memory.weights

    def update_beta(self, new_beta):
        self.combined_memory.update_beta(new_beta)
    
    def update_td_errors(self, batch_idcs, updated_td_errors):
        self.combined_memory.update_td_errors(batch_idcs, updated_td_errors)
        
class CombinedMemory(object):
    def __init__(self, agent_memory_capacity, horizon, discount_factor, p_offset, alpha, beta):
        '''
        Class to combine expert and agent memory
        '''
        self.horizon = horizon
        self.discount_factor = discount_factor
        self.beta = beta
        self.alpha = alpha
        self.memory_dict = {
            'expert':ReplayMemory(None, horizon, discount_factor, p_offset['expert'], expert=True),
            'agent':ReplayMemory(agent_memory_capacity, horizon, discount_factor, p_offset['agent'], expert=False)
        }
        self.concat_memo = np.concatenate([self.memory_dict['expert'].memory, self.memory_dict['agent'].memory])
    
    def __len__(self):
        return len(self.memory_dict['expert']) + len(self.memory_dict['agent'])
    
    def add_episode(self, obs, actions, rewards, td_errors, predictive_state, memory_id):
        #time1 = time()
        self.memory_dict[memory_id].add_episode(obs, actions, rewards, td_errors, predictive_state)
        #print(f'Time to add episode = {time() - time1:.2f}s')

        # recompute weights
        #time1 = time()
        self._update_weights()
        #print(f'Time to update weights = {time() - time1:.2f}s')
        if memory_id == 'expert': # TODO do this in a less hacky way
            self.concat_memo = self.memory_dict[memory_id].memory

        elif memory_id == 'agent':
            print(f"{len(self.memory_dict['expert'].memory) = }")
            print(f"{len(self.memory_dict['agent'].memory) = }")
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

    def __init__(self, capacity, horizon, discount_factor, p_offset, expert=False):
        self.horizon = horizon
        self.discount_factor = discount_factor
        self.p_offset = p_offset
        self.memory = deque([],maxlen=capacity)
        self.expert = int(expert)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def add_episode(self, obs, actions, rewards, td_errors, predictive_states):
        '''
        Adds all transitions within an episode to the memory.
        '''
        assert len(obs) > self.horizon, f"Expected len(obs) > self.horizon, but are {len(obs)} and {self.horizon}!"
        discount_array = np.array([self.discount_factor ** i for i in range(self.horizon)])

        for t in range(len(obs)-self.horizon):
            state = obs[t]
            next_state = obs[t+1]
            n_step_state = obs[t+self.horizon]
            predictive_state = predictive_states[t]
            next_predictive_state = predictive_states[t+1]
            n_step_predictive_state = predictive_states[t+self.horizon]
            action = actions[t]
            reward = rewards[t]
            n_step_reward = np.sum(rewards[t:t+self.horizon] * discount_array) # TODO use conv1d here?
            td_error = td_errors[t]

            self.push(
                state,
                action,
                predictive_state,
                next_state,
                next_predictive_state,
                reward,
                n_step_state,
                n_step_predictive_state,
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

def load_expert_demo(env_name, data_dir, num_expert_episodes, centroids, combined_memory, dynamics_model=None):
    
    # load data
    print(f"Loading data of {env_name}...")
    data = minerl.data.make(env_name,  data_dir=data_dir)
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)
    print(f'{len(trajectory_names) = }')

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

        if dynamics_model is not None:
            pov_obs = einops.rearrange(torch.from_numpy(np.array(list(map(lambda x: x['pov'], obs))).astype(np.float32) / 255), ' t h w c -> t c h w').to(dynamics_model.device)
            vec_obs = torch.from_numpy(np.array(list(map(lambda x: x['vector'], obs))).astype(np.float32)).to(dynamics_model.device)
            torch_actions = torch.from_numpy(centroids[np.array(actions)].astype(np.float32)).to(dynamics_model.device)
            sample, *_ = dynamics_model.visual_model.encode_only(pov_obs) 
            gru_input = torch.cat([sample, vec_obs, torch_actions], dim=1)[None]
            hidden_states_seq, _ = dynamics_model.gru(gru_input)
            predictive_state = hidden_states_seq[0]
            predictive_state = torch.cat([torch.zeros_like(predictive_state)[:1], predictive_state[:-1]], dim=0).detach().cpu().numpy()
        else:
            predictive_state = np.zeros((len(obs), 10))

        # add episode to memory
        combined_memory.add_episode(obs, actions, rewards, td_errors, predictive_state, memory_id='expert')
        print(f'Reward: {np.sum(rewards)}\n')


    print('\nLoaded ',len(combined_memory.memory_dict['expert']),' expert samples!')

    return combined_memory