import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision as tv
import minerl
import random
import numpy as np
import os
import random
import einops
from itertools import chain

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

class RewardData(Dataset):

    def __init__(self, env_name, data_dir, num_data=0, upsample=True, backfill=True, backfill_discount=0.99):
        '''
        Dataset to learn to predict reward given [pov_obs, vec_obs]
        '''
        super().__init__()

        print(f'\nLoading data of {env_name} from {data_dir}..')

        # load data
        data = np.load(os.path.join(data_dir, env_name+'_data.npz'))
        pov_obs, vec_obs, rewards, traj_starts = data['pov_obs'], data['vec_obs'], data['rewards'], data['traj_starts']

        if num_data > 0:
            pov_obs = pov_obs[:num_data]
            vec_obs = vec_obs[:num_data]
            rewards = rewards[:num_data]
            traj_starts = traj_starts[:num_data]

        self.pov_obs = np.array(pov_obs)
        self.vec_obs = np.array(vec_obs)
        self.rewards = np.array(rewards)
        self.traj_starts = np.array(traj_starts)

        self.weights = np.ones_like(self.rewards)
        num_non_zero_rew = len(self.rewards[self.rewards>0])
        num_zero_rew = len(self.rewards) - num_non_zero_rew
        print(f'Share of non-zero rewards = {num_non_zero_rew / len(self.rewards) * 100:.3f} %')
        if backfill:
            print('Backfilling... Upsampling will not be used!')
            self.rewards = self.backfill(self.rewards, self.traj_starts, backfill_discount)
        elif upsample:
            # weights of non-zero rew should have same sum as weights of zero rew
            self.weights[self.rewards > 0] = num_zero_rew / num_non_zero_rew
        
    def backfill(self, rew, starts, gamma):
        for i in range(len(rew)-2, -1, -1):
            if starts[i+1] == 1:
                rew[i] = rew[i]
            else:
                rew[i] += gamma*rew[i+1]
        return rew


    def __len__(self):
        return len(self.rewards)
    
    def __getitem__(self, idx):
        # transform image to float array
        pov = tv.transforms.functional.to_tensor(self.pov_obs[idx])

        return pov, self.vec_obs[idx].astype(np.float32), self.rewards[idx].astype(np.float32)

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

class VectorObsData(Dataset):

    def __init__(self, env_name, data_dir, num_data=0):
        super().__init__()

        print(f'\nLoading data of {env_name} from {data_dir}..')
        # load data
        data = np.load(os.path.join(data_dir, env_name+'_data.npz'))
        actions, pov_obs, vec_obs, traj_starts = data['actions'], data['pov_obs'], data['vec_obs'], data['traj_starts'] 

        new_actions = []
        new_pov_obs = []
        new_vec_obs = []
        new_targets = []

        # traverse backwards through trajectories
        n_data = 0
        cur_frame = len(traj_starts)-2
        done = False
        
        print(vec_obs[0])
        print(vec_obs[-1])
        rng = np.random.default_rng(42)
        while cur_frame >= 0:
            if cur_frame < 0:
                break
            
            # skip last frame in an episode
            if traj_starts[cur_frame+1] == 1:
                cur_frame -= 1
                continue

            if (vec_obs[cur_frame] == vec_obs[cur_frame+1]).all():
                if rng.random() > 0.025/0.975:
                    cur_frame -= 1
                    continue
            
            new_actions.append(actions[cur_frame])
            new_pov_obs.append(pov_obs[cur_frame])
            new_vec_obs.append(vec_obs[cur_frame])
            new_targets.append(0 if (vec_obs[cur_frame] == vec_obs[cur_frame+1]).all() else 1)
            
            # check if enough data has been collected
            if num_data > 0:
                n_data += 1
                if n_data >= num_data:
                    break
            cur_frame -= 1

        self.actions = np.array(new_actions)
        self.pov_obs = np.array(new_pov_obs)
        self.vec_obs = np.array(new_vec_obs)
        self.targets = np.array(new_targets)
        
        print(f'\nNumber of positive targets = {np.sum(self.targets)}')
        print(f'Percentage of positive targets = {np.sum(self.targets)/len(self.targets)}')

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        # transform image to float array
        pov = (self.pov_obs[idx].astype(np.float32) / 255)
        pov = einops.rearrange(pov, 'h w c -> c h w')

        vec_obs = self.vec_obs[idx].astype(np.float32)
        act = self.actions[idx].astype(np.float32)
        target = self.targets[idx].astype(np.float32)

        return pov, vec_obs, act, target

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
        pov_obs = einops.rearrange(np.array(pov_obs), 'b h w c -> b c h w').astype(np.float32) / 255
        vec_obs = np.array(vec_obs).astype(np.float32)
        actions = np.array([ac['vector'] for ac in actions]).astype(np.float32)
        # TODO discretize actions?

        return pov_obs, vec_obs, actions

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        return self._load_trajectory(self.names[idx])



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



