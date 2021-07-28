import torch
import dynamics_models
import minerl
import argparse
import torchvision as tv
import numpy as np
import os
from time import time

from torch.utils.tensorboard import SummaryWriter

def main(env_name, rssm_path, log_dir, data_dir):
    # set device
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    
    # load rssm model
    rssm = dynamics_models.RSSM.load_from_checkpoint(rssm_path).to(device)


    # load data
    data = minerl.data.make(env_name,  data_dir=data_dir)

    # load first trajectory
    trajectory = list(data.load_data(data.get_trajectory_names()[0], skip_interval=0, include_metadata=False))
    actions = np.array([trajectory[i][1]['vector'] for i in range(len(trajectory))]).astype(np.float32)
    rewards = np.array([trajectory[i][2] for i in range(len(trajectory))]).astype(np.float32)
    vec_obs = np.array([trajectory[i][0]['vector'] for i in range(len(trajectory))]).astype(np.float32)
    pov_obs = [trajectory[i][0]['pov'] for i in range(len(trajectory))]

    # transform to torch tensors
    pov = torch.stack([tv.transforms.functional.to_tensor(pov) for pov in pov_obs], dim=0).to(device)
    actions = torch.from_numpy(actions).to(device)
    vec = torch.from_numpy(vec_obs).to(device)
    print(f'pov.shape = {pov.shape}')
    print(f'vec.shape = {vec.shape}')
    print(f'actions.shape = {actions.shape}')

    #
    writer = SummaryWriter(log_dir=os.path.join(log_dir, str(int(time()))))
    h_n = None
    c_n = None
    for t in range(len(actions)):
        cur_pov = pov[t][None,...]
        cur_vec = vec[t][None,...]
        cur_action = actions[t][None,...]
        cur_rew = rewards[t]
        #print(f'cur_pov.shape = {cur_pov.shape}')
        #print(f'cur_vec.shape = {cur_vec.shape}')
        #print(f'cur_action.shape = {cur_action.shape}')

        (s_mean, s_std), s_t, r_t, (h_n, c_n), pov_mean, pov_std = rssm(cur_pov, cur_vec, cur_action, h_n, c_n, batched=False)
        if cur_rew == 1:
            writer.add_scalar('predicted reward', r_t, t)
        writer.add_scalar('true reward', cur_rew, t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--rssm_path', type=str, default='./')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--data_dir', type=str, default='./')
    
    args = vars(parser.parse_args())
    
    main(**args)