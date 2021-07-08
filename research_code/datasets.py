import torch
from torch.utils.data import Dataset
import torchvision as tv
import minerl
import random
import numpy as np
import os

def save_data(env_name, num_samples, num_frames, data_dir=None, num_workers=2, save_dir='./numpy_data'):
    # get data
    actions, pov_obs, vec_obs, rewards, next_pov_obs, next_vec_obs = get_data(env_name, num_samples, num_frames, data_dir, num_workers)
    kwargs = {
        'actions':actions,
        'pov_obs':pov_obs,
        'vec_obs':vec_obs,
        'rewards':rewards,
        'next_pov_obs':next_pov_obs,
        'next_vec_obs':next_vec_obs
    }

    # make sure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # save data
    save_path = save_dir + '/' + env_name + '_data.npz'
    print(f'Saving data to {save_path}..')
    np.savez_compressed(save_path, **kwargs)

def get_data(env_name, num_samples=0, num_frames=4, data_dir=None, num_workers=2):
    '''
    pass num_samples = 0 to load all available data
    '''
    if data_dir is None:
        data = minerl.data.make(env_name,  data_dir='data', num_workers=num_workers)
    else:
        data = minerl.data.make(env_name,  data_dir=data_dir, num_workers=num_workers)

    # Go over the dataset once and collect all actions and the observations (the "pov" image).
    # We do this to later on have uniform sampling of the dataset and to avoid high memory use spikes.
    all_actions = []
    all_pov_obs = []
    all_vec_obs = []
    all_rewards = []
    all_next_pov_obs = []
    all_next_vec_obs = []

    print(f"Loading data of {env_name}...")
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)

    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        trajectory = list(data.load_data(trajectory_name, skip_interval=0, include_metadata=False))
        for idx in range(len(trajectory) % num_frames, len(trajectory)-num_frames+1, num_frames): #TODO: Check that this works properly
            # extract obs, act, rew, next_obs from trajectory
            pov_obs = [trajectory[idx+i][0]['pov'] for i in range(num_frames)]
            vec_obs = [trajectory[idx+i][0]['vector'] for i in range(num_frames)]
            actions = trajectory[idx+num_frames-1][1]['vector']
            rewards = trajectory[idx+num_frames-1][2]
            next_pov_obs = [trajectory[idx+i+num_frames-1][0]['pov'] for i in range(num_frames)]
            next_vec_obs = [trajectory[idx+i+num_frames-1][0]['vector'] for i in range(num_frames)]

            # add to lists
            all_actions.append(actions)
            all_pov_obs.append(pov_obs)
            all_vec_obs.append(vec_obs)
            all_rewards.append(rewards)
            all_next_pov_obs.append(next_pov_obs)
            all_next_vec_obs.append(next_vec_obs)
        
        if len(all_actions) >= num_samples and num_samples > 0:
            break

    all_actions = np.array(all_actions)
    all_pov_obs = np.array(all_pov_obs) # transposing is handled by PILtoTensor
    all_vec_obs = np.array(all_vec_obs)
    all_rewards = np.array(all_rewards)
    all_next_pov_obs = np.array(all_next_pov_obs)
    all_next_vec_obs = np.array(all_next_vec_obs)

    return all_actions, all_pov_obs, all_vec_obs, all_rewards, all_next_pov_obs, all_next_vec_obs

class OfflineData(Dataset):

    def __init__(self, env_name, num_samples, num_frames=4, kmeans=None):

        super().__init__()

        print(f'Loading data of {env_name}..')

        # load data
        self.actions, self.pov_obs, self.vec_obs, self.rewards, self.next_pov_obs, self.next_vec_obs = get_data(env_name, num_samples, num_frames)

        # map actions to closest kmeans centroid
        if kmeans is not None:
            self.actions = kmeans.predict(self.actions)

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        # transform image to float array
        pov = torch.stack([tv.transforms.functional.to_tensor(pic) for pic in self.pov_obs[idx]], dim=0).squeeze()
        next_pov = torch.stack([tv.transforms.functional.to_tensor(pic) for pic in self.next_pov_obs[idx]], dim=0).squeeze()

        return self.actions[idx].astype(np.int64), pov, self.vec_obs[idx], self.rewards[idx], next_pov, self.next_vec_obs[idx]