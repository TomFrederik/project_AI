import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import minerl

def main(env_name, data_dir, centroids_path, num_centroids, out_dir):

    # make sure out dir exists
    os.makedirs(out_dir, exist_ok=True)

    # load centroids
    centroids_path = os.path.join(centroids_path, env_name + '_' + str(num_centroids) + '_centroids.npy')
    print(f'\nLoading centroids from {centroids_path}...')
    centroids = np.load(centroids_path)


    pipeline = minerl.data.make(env_name, data_dir)
    #pipeline = minerl.data.make('MineRLObtainIronPickaxeVectorObf-v0', data_dir)
    all_names = pipeline.get_trajectory_names()
    all_actions = []
    for name in tqdm(all_names):
        traj_data = pipeline.load_data(name)

        # unpack data
        _, actions, *_ = zip(*traj_data)
        actions = np.array([ac['vector'] for ac in actions]).astype(np.float32)

        # compute action clusters
        actions = np.argmin(((centroids[None,:,:] - actions[:,None,:]) ** 2).sum(axis=-1), axis=1).astype(np.int64)

        all_actions.append(actions)
    
    all_actions = np.concatenate(all_actions, axis=0)
    unique, counts = np.unique(all_actions, return_counts=True)

    plt.figure()
    plt.bar(np.arange(len(unique)), np.log(counts)+1, width=1)
    plt.ylabel('log(count)+1')
    plt.xlabel('Action')
    plt.title(env_name[6:] + ' - ' + str(num_centroids) + ' centroids')
    plt.savefig(os.path.join(out_dir, env_name[6:] + ' - ' + str(num_centroids) + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--centroids_path', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--num_centroids', type=str, default=100)
    parser.add_argument('--out_dir', type=str, default='/home/lieberummaas/datadisk/minerl/plots')

    args = parser.parse_args()

    main(**vars(args))