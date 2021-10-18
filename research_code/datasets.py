import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision as tv
import minerl
import random
import numpy as np
import os
import random
import einops
from random import shuffle
from itertools import chain
from collections import deque
from time import time

ENVS = ['MineRLObtainIronPickaxeDenseVectorObf-v0', 'MineRLObtainDiamondDenseVectorObf-v0',
        'MineRLTreechopVectorObf-v0', 'MineRLObtainDiamondVectorObf-v0', 'MineRLObtainIronPickaxeVectorObf-v0']

        
def extract_data(env_name, num_samples=0, data_dir=None, save_dir='./numpy_data'):
    
    # make sure data_dir exists
    if data_dir is None:
        data_dir = './data/'
    
    print('Data dir in extract data is ', data_dir)

    # make sure data_dir exists
    os.makedirs(os.path.join(data_dir, env_name), exist_ok=True)
    
    # get data
    actions, pov_obs, vec_obs, rewards, traj_starts = get_data(env_name, num_samples, data_dir)
    kwargs = {
        'actions':actions,
        'pov_obs':pov_obs,
        'vec_obs':vec_obs,
        'rewards':rewards,
        'traj_starts':traj_starts
    }

    # make sure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # add env identifier
    save_path = os.path.join(save_dir, env_name + '_data.npz')
    print(f'Saving data to {save_path}..')
    
    # save data
    np.savez_compressed(save_path, **kwargs)


def get_data(env_name, num_samples=0, data_dir=None):
    '''
    pass num_samples = 0 to load all available data
    '''
    # download data
    #minerl.data.download(environment=env_name, directory=data_dir)
    
    # load data
    data = minerl.data.make(env_name,  data_dir=data_dir)
    
    # Go over the dataset once and collect all actions and the observations (the "pov" image).
    # We do this to later on have uniform sampling of the dataset and to avoid high memory use spikes.
    all_actions = []
    all_pov_obs = []
    all_vec_obs = []
    all_rewards = []
    all_traj_starts = [] # [1, 0, 0, 0, ..., 1, 0, 0, ...]

    print(f"Loading data of {env_name}...")
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)

    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        # load trajectory
        trajectory = list(data.load_data(trajectory_name, skip_interval=0, include_metadata=False))

        # add to lists
        all_actions.extend([trajectory[i][1]['vector'] for i in range(len(trajectory))])
        all_rewards.extend([trajectory[i][2] for i in range(len(trajectory))])
        all_pov_obs.extend([trajectory[i][0]['pov'] for i in range(len(trajectory))])
        all_vec_obs.extend([trajectory[i][0]['vector'] for i in range(len(trajectory))])
        all_traj_starts.extend([1] + [0]*(len(trajectory)-1))
        if len(all_actions) >= num_samples and num_samples > 0:
            break

    all_actions = np.array(all_actions)
    all_pov_obs = np.array(all_pov_obs) # transposing is handled by PILtoTensor
    all_vec_obs = np.array(all_vec_obs)
    all_rewards = np.array(all_rewards)
    all_traj_starts = np.array(all_traj_starts)

    return all_actions, all_pov_obs, all_vec_obs, all_rewards, all_traj_starts


class TrajectoryData(Dataset):

    def __init__(self, env_name, data_dir):
        '''
        Dataset that returns whole trajectories
        '''
        super().__init__()

        # load data
        self.data = minerl.data.make(env_name, data_dir)
        self.names = self.data.get_trajectory_names()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # load traj
        obs, act, rew, *_ = zip(*self.data.load_data(self.names[idx]))

        # convert to np float 32
        vec_obs = np.array([o['vector'] for o in obs]).astype(np.float32)
        act = np.array([a['vector'] for a in act]).astype(np.float32)
        pov_obs = np.array([o['pov'] for o in obs]).astype(np.float32) / 255
        pov_obs = einops.rearrange(pov_obs, 'b h w c -> b c h w')
        rew = np.array(rew).astype(np.float32)
        
        return pov_obs, vec_obs, act, rew

class DynamicsData(IterableDataset):
    def __init__(self, env_name, data_dir, seq_len, batch_size):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pipeline = minerl.data.make(env_name, data_dir)
        self.buffer = deque([])
        self.names = list(self.pipeline.get_trajectory_names())
        shuffle(self.names)
        
    def _load_new_traj(self, name):
        obs, act, rew, *_ = zip(*self.pipeline.load_data(name))
        
        pov, vec = zip(*[(o['pov'], o['vector']) for o in obs])
        pov = np.stack(pov)
        vec = np.stack(vec)

        pov = einops.rearrange(pov.astype(np.float32), 't h w c -> t c h w') / 255
        vec = vec.astype(np.float32)
        act = np.stack([a['vector'] for a in act]).astype(np.float32)
        rew = np.array([r for r in rew]).astype(np.float32)
        
        start_idcs = [len(obs) % self.seq_len + self.seq_len * i for i in range(len(obs)//self.seq_len)]
        pov_list = [pov[start_idx:start_idx+self.seq_len] for start_idx in start_idcs]
        vec_list = [vec[start_idx:start_idx+self.seq_len] for start_idx in start_idcs]
        act_list = [act[start_idx:start_idx+self.seq_len] for start_idx in start_idcs]
        rew_list = [rew[start_idx:start_idx+self.seq_len] for start_idx in start_idcs]
    
        self.buffer.extend(zip(pov_list, vec_list, act_list, rew_list))
        shuffle(self.buffer)
    
    def _iterator(self):
        cur_name_idx = 0
        # yield as long as there are samples in the buffer or new samples that can still be loaded
        while len(self.buffer) > 0 or cur_name_idx < len(self.names):            
            # load a new trajectory when the buffer is running out
            if len(self.buffer) < self.batch_size and cur_name_idx < len(self.names):
                self._load_new_traj(self.names[cur_name_idx])
                cur_name_idx += 1
        
            # get a new sample from buffer
            pov, vec, act, rew = self.buffer.popleft()

            # add a batch dimension        
            yield pov, vec, act, rew
    
    def __iter__(self):
        return self._iterator()

class SingleSequenceDynamics(Dataset):
    def __init__(self, env_name, data_dir, seq_len, batch_size):
        self.dataset = DynamicsData(env_name, data_dir, seq_len, batch_size)
        self.batch_size = batch_size
        iterator = iter(self.dataset)
        self.seq_1 = next(iterator)
        self.seq_2 = next(iterator)
    
    def __len__(self):
        return 2
    
    def __getitem__(self, idx):
        #return self.batch
        if idx == 0:
            return [x[None] for x in self.seq_1]
        elif idx == 1:
            return [x[None] for x in self.seq_2]


class PretrainQNetIterableData(IterableDataset):
    def __init__(self, env_name, data_dir, centroids, n_step, gamma, num_workers):
        super().__init__()
        
        self.n_step = n_step
        self.gamma = gamma
        self.centroids = centroids
        self.num_workers = num_workers
        self.pipeline = minerl.data.make(env_name, data_dir)
        self.names = self.pipeline.get_trajectory_names()
        random.shuffle(self.names)

        # split trajectories between workers
        if self.num_workers > 0:
            trajectories_per_worker = len(self.names) // self.num_workers
            self.names_per_worker = {
                worker_id: self.names[trajectories_per_worker * worker_id:trajectories_per_worker*(worker_id+1)] for worker_id in range(self.num_workers)
            }
        
        # compute discount array
        self.discount_array = np.array([self.gamma ** i for i in range(self.n_step)])
        
    def _load_trajectory(self, name):
        # load trajectory data
        data = self.pipeline.load_data(name)
        
        # unpack data
        obs, actions, rewards, *_ = zip(*data)
        pov_obs, vec_obs = [item['pov'] for item in obs], [item['vector'] for item in obs]
        pov_obs = einops.rearrange(np.array(pov_obs), 'b h w c -> b c h w').astype(np.float32) / 255
        vec_obs = np.array(vec_obs).astype(np.float32)
        #rewards = np.log(1 + np.array(rewards).astype(np.float32)) # log for better scaled rewards
        rewards = np.array(rewards).astype(np.float32)
        actions = np.array([ac['vector'] for ac in actions]).astype(np.float32)

        # compute actions
        actions = np.argmin(((self.centroids[None,:,:] - actions[:-self.n_step,None,:]) ** 2).sum(axis=-1), axis=1).astype(np.int64)
        
        return pov_obs, vec_obs, actions, rewards

    def _get_trajectory_iterator(self, name):
        pov_obs, vec_obs, actions, rewards = self._load_trajectory(name)
        idcs = np.random.permutation(len(actions))
        for idx in idcs:
            yield pov_obs[idx], vec_obs[idx], actions[idx], rewards[idx], pov_obs[idx+1], vec_obs[idx+1], np.sum(rewards[idx:idx+self.n_step]*self.discount_array).astype(np.float32), pov_obs[idx+self.n_step], vec_obs[idx+self.n_step]
    
    def _get_stream_of_trajectories(self, names):
        return chain.from_iterable(map(self._get_trajectory_iterator, names))
        
    def __iter__(self):
        if self.num_workers == 0:
            return self._get_stream_of_trajectories(self.names)
        else:
            worker_id = torch.utils.data.get_worker_info().id
            return self._get_stream_of_trajectories(self.names_per_worker[worker_id])
        
class TrajectoryIterData(IterableDataset):
    def __init__(self, env_name, data_dir, num_episodes=0, centroids=None, num_workers=0):
        super().__init__()

        self.centroids = centroids
        self.num_workers = num_workers
        self.pipeline = minerl.data.make(env_name, data_dir)
        self.names = self.pipeline.get_trajectory_names()
        self.names.sort()
        
        blacklist = [
            'v3_right_basil_dragon-15_4328-4804',
            'v3_kindly_lemon_mummy-2_59830-60262',
            'v3_right_basil_dragon-15_281-899',
            'v3_absolute_grape_changeling-47_826-1734',
            'v3_right_basil_dragon-15_6324-6873'
        ]
        for name in blacklist:
            self.names.remove(name)

        if num_episodes > 0:
            self.names = self.names[:num_episodes]

        # split trajectories between workers
        if self.num_workers > 0:
            trajectories_per_worker = len(self.names) // self.num_workers
            self.names_per_worker = {
                worker_id: self.names[trajectories_per_worker * worker_id:trajectories_per_worker*(worker_id+1)] for worker_id in range(self.num_workers)
            }
        
    def _load_trajectory(self, name):
        print(f'Loading trajectory {name}..')
        # load trajectory data
        data = self.pipeline.load_data(name)
        
        # unpack data
        obs, actions, rewards, *_ = zip(*data)
        pov_obs, vec_obs = [item['pov'] for item in obs], [item['vector'] for item in obs]
        pov_obs = einops.rearrange(np.array(pov_obs), 't h w c -> t c h w').astype(np.float32) / 255
        vec_obs = np.array(vec_obs).astype(np.float32)
        actions = np.array([ac['vector'] for ac in actions]).astype(np.float32)
        rewards = np.array(rewards).astype(np.float32)
        
        # EXPERIMENTAL
        print('Warning, setting last reward to 0')
        rewards[-1] = 0
        
        if self.centroids is not None:
            # compute action idcs
            action_idx = np.argmin(((self.centroids[None,:,:] - actions[:,None,:]) ** 2).sum(axis=-1), axis=1).astype(np.int64)
            return pov_obs, vec_obs, actions, action_idx, rewards
        else:
            return pov_obs, vec_obs, actions, rewards

    def _get_stream_of_trajectories(self, names):
        return map(self._load_trajectory, names)
        
    def __iter__(self):
        if self.num_workers == 0:
            return self._get_stream_of_trajectories(self.names)
        else:
            worker_id = torch.utils.data.get_worker_info().id
            return self._get_stream_of_trajectories(self.names_per_worker[worker_id])


class TrajectoryData(Dataset):
    def __init__(self, env_name, data_dir, num_episodes=0, centroids=None):
        super().__init__()

        self.centroids = centroids
        self.pipeline = minerl.data.make(env_name, data_dir)
        self.names = self.pipeline.get_trajectory_names()
        self.names.sort()
        if num_episodes > 0:
            self.names = self.names[:num_episodes]

        
    def _load_trajectory(self, name):
        # load trajectory data
        data = self.pipeline.load_data(name)
        
        # unpack data
        obs, actions, rewards, *_ = zip(*data)
        pov_obs, vec_obs = [item['pov'] for item in obs], [item['vector'] for item in obs]
        pov_obs = einops.rearrange(np.array(pov_obs), 't h w c -> t c h w').astype(np.float32) / 255
        vec_obs = np.array(vec_obs).astype(np.float32)
        actions = np.array([ac['vector'] for ac in actions]).astype(np.float32)
        rewards = np.array(rewards).astype(np.float32)
        
        if self.centroids is not None:
            # compute action idcs
            action_idx = np.argmin(((self.centroids[None,:,:] - actions[:,None,:]) ** 2).sum(axis=-1), axis=1).astype(np.int64)
            return pov_obs, vec_obs, actions, action_idx, rewards
        else:
            return pov_obs, vec_obs, actions, rewards


    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        print(f'Loading trajectory {self.names[idx]}..')
        return self._load_trajectory(self.names[idx])





class StateVQVAEData(Dataset):
    def __init__(self, env_name, data_dir, num_workers, num_trajs):
        super().__init__()
        
        self.max_len = 20000
        self.num_trajs = num_trajs
        self.num_workers = num_workers
        self.pipeline = minerl.data.make(env_name, data_dir)
        self.names = self.pipeline.get_trajectory_names()
        if self.num_trajs > 0:
            self.names = self.names[:self.num_trajs]

    def _load_trajectory(self, name):
        # load trajectory data
        data = self.pipeline.load_data(name)
        
        # unpack data
        obs, actions, *_ = zip(*data)
        pov_obs, vec_obs = [item['pov'] for item in obs], [item['vector'] for item in obs]
        pov_obs = einops.rearrange(np.array(pov_obs), 't h w c -> t c h w').astype(np.float32) / 255
        vec_obs = np.array(vec_obs).astype(np.float32)
        actions = np.array([ac['vector'] for ac in actions]).astype(np.float32)

        return pov_obs[:self.max_len], vec_obs[:self.max_len], actions[:self.max_len]

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        print(f'Loading trajectory {self.names[idx]}..')
        return self._load_trajectory(self.names[idx])



class BufferedBatchDataset(IterableDataset):
    '''
    For docs on BufferedBatchIter, see https://github.com/minerllabs/minerl/blob/dev/minerl/data/buffered_batch_iter.py
    '''
    def __init__(self, env_name, data_dir, batch_size, num_epochs):
        # save params
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # create data pipeline
        self.data = minerl.data.make(env_name, data_dir=data_dir)
        
        # create iterator from pipeline
        self.iter = minerl.data.BufferedBatchIter(self.data)       
         
    def __iter__(self):
        '''
        Returns next pov_obs in the iterator.
        '''
        return self.iter.buffered_batch_iter(self.batch_size, self.num_epochs)






