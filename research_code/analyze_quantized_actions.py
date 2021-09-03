from action_vqvae import ActionVQVAE
import argparse
import os
import minerl
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def main(
    env_name,
    save_dir,
    action_quantizer
):

    # set data path
    data_path = os.path.join(save_dir, env_name, f'actions_version_{action_quantizer}.npz')

    # load data
    data = np.load(data_path)
    deobf_data = data['unique_deobf']
    obf_data = data['all_obf']

    # 
    unique_quantized = pd.DataFrame(obf_data).drop_duplicates().to_numpy()

    # check if there is at least one collision
    if len(deobf_data) == len(unique_quantized):
        print("\n'deobf_data' and 'unqiue_quantized' have the same number of elements!")
        print('That means there is no possible collision!')
        print('Terminating early, without plotting...')
        return
    else:
        print(f'Number of unique actions = {len(deobf_data)}')
        print(f'Number of unique clusters = {len(unique_quantized)}')

    # for every unique quantized action, find all deobf actions which get mapped to that quantized action
    cluster_id_to_deobf = {}
    for i in tqdm(range(len(unique_quantized))):
        cluster_id_to_deobf[i] = deobf_data[(obf_data == unique_quantized[i]).all(axis=1)]
        #print(np.std(cluster_id_to_deobf[i], axis=0))
        #print(cluster_id_to_deobf[i])
        #fig = plt.figure(figsize=(6,100))
        #print(np.min(1 + np.abs(np.min(cluster_id_to_deobf[i])) + cluster_id_to_deobf[i]))
        #plt.imshow(np.log(1 + np.abs(np.min(cluster_id_to_deobf[i])) + cluster_id_to_deobf[i]), cmap='viridis', interpolation='nearest')
        #plt.savefig(os.path.join(save_dir, env_name, f'cluster_0_version_{action_quantizer}.png'))
        #raise ValueError
    #print(cluster_id_to_deobf)

    # plot number of collisions
    num_collisions = [v.shape[0] for v in cluster_id_to_deobf.values()]
    fig = plt.figure()
    plt.bar(np.arange(len(num_collisions)), num_collisions)
    plt.ylabel('Collisions')
    plt.xlabel('Action ID')
    plt.savefig(os.path.join(save_dir, env_name, f'collisions_version_{action_quantizer}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechop-v0')
    parser.add_argument('--save_dir', type=str, default='/home/lieberummaas/datadisk/minerl/action_analysis')
    parser.add_argument('--action_quantizer', type=int, default=0)
    
    args = parser.parse_args()
    
    main(**vars(args))