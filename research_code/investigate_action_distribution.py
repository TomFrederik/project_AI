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
        for ac in actions:
            all_actions.append(ac['vector'])
    
    all_actions = np.array(all_actions)
    
    mean = np.mean(all_actions, axis=0)
    std = np.std(all_actions, axis=0)

    print(f'Mean = {mean}')
    print(f'Std = {std}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--num_trajs', type=int, default=0)
    parser.add_argument('--eps', type=float, default=1e-8)

    args = parser.parse_args()

    main(**vars(args))