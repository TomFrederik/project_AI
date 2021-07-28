import torch
from torch.utils.data import Dataset
import torchvision as tv
import minerl
import random
import numpy as np
import os
import random

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

class VAEData(Dataset):

    def __init__(self, env_name, data_dir, num_data=0):

        super().__init__()

        print(f'\nLoading data of {env_name} from {data_dir}..')

        # load data
        data = np.load(os.path.join(data_dir, env_name+'_data.npz'))
        actions, pov_obs, vec_obs, rewards, traj_starts = data['actions'], data['pov_obs'], data['vec_obs'], data['rewards'], data['traj_starts'] 

        if num_data > 0:
            actions = actions[:num_data]
            pov_obs = pov_obs[:num_data]
            vec_obs = vec_obs[:num_data]
            rewards = rewards[:num_data]

        self.actions = np.array(actions)
        self.pov_obs = np.array(pov_obs)
        self.vec_obs = np.array(vec_obs)
        self.rewards = np.array(rewards)


    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        # transform image to float array
        pov = tv.transforms.functional.to_tensor(self.pov_obs[idx])

        return self.actions[idx].astype(np.float32), pov, self.vec_obs[idx], self.rewards[idx]


class DynamicsData(Dataset):

    def __init__(self, env_name, data_dir, seq_len=4, num_data=0):
        '''
        Will load seq_len sequences, first for input, rest as target
        longer seq len doesn't make sense since it would need to know the actions at each timestept too
        '''

        super().__init__()

        print(f'\nLoading data of {env_name} from {data_dir}..')
        # load data
        data = np.load(os.path.join(data_dir, env_name+'_data.npz'))
        actions, pov_obs, vec_obs, rewards, traj_starts = data['actions'], data['pov_obs'], data['vec_obs'], data['rewards'], data['traj_starts'] 

        num_frames = seq_len

        new_actions = []
        new_pov_obs = []
        new_vec_obs = []
        new_rewards = []

        # traverse backwards through trajectories
        n_data = 0
        cur_frame = len(traj_starts)-1-num_frames
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
            else:
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
        pov = torch.stack([tv.transforms.functional.to_tensor(pic) for pic in self.pov_obs[idx]], dim=0).squeeze()
        pov = pov.numpy().astype(np.float32)

        vec_obs = self.vec_obs[idx].astype(np.float32)
        act = self.actions[idx].astype(np.float32)
        rew = self.rewards[idx].astype(np.float32)

        return pov, vec_obs, act, rew


class BehavCloneData(Dataset):

    def __init__(self, env_name, data_dir):

        super().__init__()
        
        print(f'\nLoading data of {env_name} from {data_dir}..')
        # load data
        data = np.load(os.path.join(data_dir, env_name+'_data.npz'))
        self.actions, self.pov_obs, self.vec_obs = data['actions'], data['pov_obs'], data['vec_obs']

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        # transform image to float array
        pov = tv.transforms.functional.to_tensor(self.pov_obs[idx])
        pov = pov.numpy().astype(np.float32)

        # get action
        act = self.actions[idx].astype(np.float32)

        return pov, self.vec_obs[idx].astype(np.float32), act


class TrajData(Dataset):

    def __init__(self, env_name, data_dir, num_workers, num_frames):

        self.data = minerl.data.make(env_name, data_dir, num_workers)
        self.traj_names = data.get_trajectory_names()
        random.shuffle(self.traj_names)

        # get next traj
        self.cur_traj = next(self._data_generator)
        self.pointer = len(self.cur_traj)-num_frames

    def __len__(self):
        pass
    '''
    def __getitem__(self, idx):
        sar = self.cur_traj[self.pointer:self.pointer+self.num_frames]
        pov = sar[0]['pov']
        vec = sar[0]['vec']
        act = sar[1]['vector']
        rew = sar[2]
        #TODO currently not using idx --> am shuffling trajectories in beginning but not batches..
        # update pointer
        self.pointer = self.pointer - self.num_frames

        return pov, vec, act


    def _data_generator(self):
        for name in self.traj_names:
            traj_data = self.data.load_data(name, skip_interval=0, include_metadata=False)
            yield traj_data
    '''
