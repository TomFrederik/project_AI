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

        
def extract_data(env_name, num_samples, data_dir=None, save_dir='./numpy_data'):
    
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


class RewardData(IterableDataset):

    def __init__(self, env_name, data_dir, num_workers=1, upsample=False, backfill=False, backfill_discount=0.99):
        '''
        Dataset to learn to predict reward given vec_obs
        '''
        super().__init__()

        # load data
        self.data = minerl.data.make(env_name, data_dir)
        
        self.names = self.data.get_trajectory_names()
        shuffle(self.names)
        worker_names = [self.names[i:len(self.names)//num_workers*(i+1)] for i in range(num_workers)]
        self.iter_per_worker = [iter(worker_names[i]) for i in range(num_workers)]
    def backfill(self, rew, starts, gamma):
        for i in range(len(rew)-2, -1, -1):
            if starts[i+1] == 1:
                rew[i] = rew[i]
            else:
                rew[i] += gamma*rew[i+1]
        return rew

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id

        # load traj
        obs, _, rew, *_ = zip(*self.data.load_data(next(self.iter_per_worker[worker_id])))

        # convert to np float 32
        vec_obs = np.array([o['vector'] for o in obs]).astype(np.float32)
        rew = np.array(rew).astype(np.float32)
        
        yield vec_obs, rew


'''
class DynamicsData(Dataset):

    def __init__(self, env_name, data_dir, seq_len=4, num_data=0):

        super().__init__()

        print(f'\nLoading data of {env_name} from {data_dir}..')
        # load data
        data = np.load(os.path.join(data_dir, env_name+'_data.npz'))
        actions, pov_obs, vec_obs, rewards, traj_starts = data['actions'], data['pov_obs'], data['vec_obs'], data['rewards'], data['traj_starts'] 

        num_frames = seq_len
        print(f'\nNumber of frames (conditioning + sequence) = {num_frames}')
        
        new_actions = []
        new_pov_obs = []
        new_vec_obs = []
        new_rewards = []

        # traverse backwards through trajectories
        n_data = 0
        cur_frame = len(traj_starts)-1-num_frames
        done = False
        while cur_frame >= 0:
            if cur_frame < 0:
                break
            
            # don't skip last frames in an episode
            if 1 in traj_starts[cur_frame:cur_frame+num_frames]:
                for i in range(num_frames):
                    if traj_starts[cur_frame + i] == 1:
                        cur_frame = cur_frame + i - num_frames
                        break
                continue
            
            new_actions.append(actions[cur_frame:cur_frame+num_frames])
            new_pov_obs.append(pov_obs[cur_frame:cur_frame+num_frames])
            new_vec_obs.append(vec_obs[cur_frame:cur_frame+num_frames])
            new_rewards.append(rewards[cur_frame:cur_frame+num_frames])
            
            # check if enough data has been collected
            if num_data > 0:
                n_data += 1
                if n_data >= num_data:
                    break

            # decrease frames
            cur_frame -= num_frames

        self.actions = np.array(new_actions)
        self.pov_obs = np.array(new_pov_obs)
        self.vec_obs = np.array(new_vec_obs)
        self.rewards = np.array(new_rewards)

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        # transform image to float array
        pov = (self.pov_obs[idx].astype(np.float32) / 255)
        pov = einops.rearrange(pov, 't h w c -> t c h w')

        vec_obs = self.vec_obs[idx].astype(np.float32)
        act = self.actions[idx].astype(np.float32)
        rew = self.rewards[idx].astype(np.float32)

        return pov, vec_obs, act, rew
'''

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
        rewards = np.log(1 + np.array(rewards).astype(np.float32)) # log for better scaled rewards
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
        
       
class PretrainQNetData(Dataset):

    def __init__(self, env_name, data_dir, centroids, n, gamma, num_data=0):
        super().__init__()

        print(f'\nLoading data of {env_name} from {data_dir}..')
        # load data
        data = np.load(os.path.join(data_dir, env_name+'_data.npz'))
        pov_obs, vec_obs, actions, rewards, traj_starts = data['pov_obs'], data['vec_obs'], data['actions'], data['rewards'], data['traj_starts'] 

        pov_obs_list = []
        vec_obs_list = []
        actions_list = []
        rewards_list = []
        next_pov_obs_list = []
        next_vec_obs_list = []
        n_step_rewards_list = []
        n_step_pov_obs_list = []
        n_step_vec_obs_list = []

        discount_array = np.array([gamma ** i for i in range(n)])

        # traverse backwards through trajectories
        n_data = 0
        cur_frame = len(traj_starts)-1-n
        done = False
        while cur_frame >= 0:
            if cur_frame < 0:
                break
            
            # skip last frames in an episode
            # TODO make this better, so that you still use the last frames?
            if 1 in traj_starts[cur_frame:cur_frame + n]:
                for i in range(n):
                    if traj_starts[cur_frame + i] == 1:
                        cur_frame = cur_frame + i - n
                        break
                continue
            
            pov_obs_list.append(pov_obs[cur_frame])            
            vec_obs_list.append(vec_obs[cur_frame])
            actions_list.append(np.argmin(((centroids - actions[cur_frame][None,:]) ** 2).sum(axis=1)))
            rewards_list.append(rewards[cur_frame])
            next_pov_obs_list.append(pov_obs[cur_frame+1])            
            next_vec_obs_list.append(vec_obs[cur_frame+1])
            n_step_rewards_list.append(np.sum(rewards[cur_frame:cur_frame+n] * discount_array))
            n_step_pov_obs_list.append(pov_obs[cur_frame+n])            
            n_step_vec_obs_list.append(vec_obs[cur_frame+n])
            
            # check if enough data has been collected
            if num_data > 0:
                n_data += 1
                if n_data >= num_data:
                    break
            cur_frame -= 1

        self.pov_obs = np.array(pov_obs_list)
        self.vec_obs = np.array(vec_obs_list)
        self.actions = np.array(actions_list)
        self.rewards = np.array(rewards_list)
        self.next_pov_obs = np.array(next_pov_obs_list)
        self.next_vec_obs = np.array(next_vec_obs_list)
        self.n_step_rewards = np.array(n_step_rewards_list)
        self.n_step_pov_obs = np.array(n_step_pov_obs_list)
        self.n_step_vec_obs = np.array(n_step_vec_obs_list)

    def __len__(self):
        return len(self.pov_obs)
    
    def __getitem__(self, idx):
        # transform image to float array
        pov = (self.pov_obs[idx].astype(np.float32) / 255)
        pov = einops.rearrange(pov, 'h w c -> c h w')
        
        next_pov = (self.next_pov_obs[idx].astype(np.float32) / 255)
        next_pov = einops.rearrange(next_pov, 'h w c -> c h w')
        
        n_step_pov = (self.n_step_pov_obs[idx].astype(np.float32) / 255)
        n_step_pov = einops.rearrange(n_step_pov, 'h w c -> c h w')
        
        vec_obs = self.vec_obs[idx].astype(np.float32)
        next_vec_obs = self.next_vec_obs[idx].astype(np.float32)
        n_step_vec_obs = self.n_step_vec_obs[idx].astype(np.float32)
        
        action = self.actions[idx].astype(np.int64)
        
        reward = self.rewards[idx].astype(np.float32)
        n_step_reward = self.n_step_rewards[idx].astype(np.float32)
        
        return pov, vec_obs, action, reward, next_pov, next_vec_obs, n_step_reward, n_step_pov, n_step_vec_obs

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
        # TODO discretize actions?

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






