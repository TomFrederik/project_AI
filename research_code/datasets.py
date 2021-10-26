from collections import deque
from itertools import chain
import os
import random
from random import shuffle
from time import time

import einops
import minerl
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision as tv

        
class TrajectoryIterData(IterableDataset):
    def __init__(self, env_name, data_dir, num_episodes=0, centroids=None, num_workers=0):
        super().__init__()

        self.centroids = centroids
        self.num_workers = num_workers
        self.pipeline = minerl.data.make(env_name, data_dir)
        self.names = self.pipeline.get_trajectory_names()
        self.names.sort()
        
        # there is something off with these trajectories -> extremely large spike in TD-error
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
        # print('Warning, setting last reward to 1')
        # rewards[-1] = 1
        
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






