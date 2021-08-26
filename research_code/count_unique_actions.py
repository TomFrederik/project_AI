import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import minerl

def main(env_name, data_dir, num_trajs, eps):
    pipeline = minerl.data.make(env_name, data_dir)
    all_names = pipeline.get_trajectory_names()

    all_actions = []
    num_actions = 0

    for i, name in tqdm(enumerate(all_names)):
        if num_trajs > 0:
            if i >= num_trajs:
                break 
        traj_data = pipeline.load_data(name)

        # unpack data
        _, actions, _, _, _ = zip(*traj_data)
        num_actions += len(actions)
        traj_actions = np.vstack([ac['vector'] for ac in actions])
        if len(all_actions) == 0:
            #print(traj_actions[0].shape)
            all_actions.append(traj_actions[0])
            traj_actions = traj_actions[1:,:]
        #print(np.vstack(all_actions)[:,None,:].shape)
        #print(traj_actions.shape)
        #print(traj_actions[None,:,:].shape)
        diff = np.sum((np.vstack(all_actions)[:,None,:] - traj_actions[None,:,:])**2, axis=-1) # (num_all_actions, num_traj_actions)
        #print(diff.shape)
        #print(diff)
        mindiff = np.min(diff, axis=0)
        #print(mindiff)
        all_actions.extend(list(traj_actions[mindiff > eps]))

        print(f'There are {len(all_actions)} unique actions out of {num_actions} total actions.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--num_trajs', type=int, default=0)
    parser.add_argument('--eps', type=float, default=1e-8)

    args = parser.parse_args()

    main(**vars(args))