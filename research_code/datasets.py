import torch
from torch.utils.data import Dataset
import torchvision as tv
import minerl
import random
import numpy as np
import os

ENVS = ['MineRLTreechopVectorObf-v0', 'MineRLNavigateVectorObf-v0',
        'MineRLNavigateDenseVectorObf-v0', 'MineRLNavigateExtremeVectorObf-v0',
        'MineRLNavigateExtremeDenseVectorObf-v0', 'MineRLObtainDiamondVectorObf-v0',
        'MineRLObtainDiamondDenseVectorObf-v0', 'MineRLObtainIronPickaxeVectorObf-v0',
        'MineRLObtainIronPickaxeDenseVectorObf-v0']

        
def extract_data(env_name, num_samples, data_dir=None, save_dir='./numpy_data'):
    
    # make sure data_dir exists
    if data_dir is None:
        data_dir = './data/'
    
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

class OfflineData(Dataset):

    def __init__(self, env_name, data_dir, num_frames=4):

        super().__init__()

        print(f'Loading data of {env_name} from {data_dir}..')

        # load data
        data = np.load(os.path.join(data_dir, env_name+'_data.npz'))
        actions, pov_obs, vec_obs, rewards, traj_starts = data['actions'], data['pov_obs'], data['vec_obs'], data['rewards'], data['traj_starts'] 

        #TODO do frame extraction
        self.num_frames = num_frames

        new_actions = []
        new_pov_obs = []
        new_vec_obs = []
        new_rewards = []

        # traverse backwards through trajectories
        for i in range(len(traj_starts)-1-2*num_frames, 0, -num_frames):
            # skip if we would cross episodes
            if 1 in traj_starts[i:i+2*num_frames]:
                # TODO: DO NOT SKIP LAST FRAMES IN EARLIER EPISODE
                continue
            new_actions.append(actions[i:i+num_frames]) # TODO: IS THIS CORRECT INDEXING?
            new_pov_obs.append(pov_obs[i:i+2*num_frames])
            new_vec_obs.append(vec_obs[i:i+2*num_frames])
            new_rewards.append(rewards[i:i+num_frames])

        self.actions = np.array(new_actions)
        self.pov_obs = np.array(new_pov_obs)
        self.vec_obs = np.array(new_vec_obs)
        self.rewards = np.array(new_rewards)


    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        # transform image to float array
        pov = torch.stack([tv.transforms.functional.to_tensor(pic) for pic in self.pov_obs[idx,:self.num_frames]], dim=0).squeeze()
        next_pov = torch.stack([tv.transforms.functional.to_tensor(pic) for pic in self.pov_obs[idx,self.num_frames:]], dim=0).squeeze()

        return self.actions[idx].astype(np.int64), pov, self.vec_obs[idx,:self.num_frames], self.rewards[idx], next_pov, self.vec_obs[idx,self.num_frames:]