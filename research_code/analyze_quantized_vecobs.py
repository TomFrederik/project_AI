from vecobs_vqvae import VecObsVQVAE
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
    vecobs_quantizer
):

    # set data path
    data_path = os.path.join(save_dir, env_name, f'obs_version_{vecobs_quantizer}.npz')

    # load data
    data = np.load(data_path)
    deobf_data = data['unique_deobf']
    obf_data = data['all_obf']

    # 
    unique_quantized = pd.DataFrame(obf_data).drop_duplicates().to_numpy()

    # check if there is at least one collision
    print(f'Number of unique obs = {len(deobf_data)}')
    print(f'Number of unique clusters = {len(unique_quantized)}')
    if len(deobf_data) == len(unique_quantized):
        print("\n'deobf_data' and 'unqiue_quantized' have the same number of elements!")
        print('That means there is no possible collision!')
        print('Terminating early, without plotting...')
        return
        
    # for every unique quantized vecobs, find all deobf obs which get mapped to that quantized vecobs
    cluster_to_deobf = [deobf_data[(obf_data == quant).all(axis=1)] for quant in tqdm(unique_quantized)]
    num_collisions = np.array([v.shape[0] for v in cluster_to_deobf])
    sorting_idcs = np.argsort(num_collisions)
    cluster_to_deobf = [cluster_to_deobf[i] for i in np.argsort(num_collisions)]

    # plot standard deviation within one cluster
    fig = plt.figure(figsize=(6,100))
    stds = [np.std(cluster, axis=0) for cluster in cluster_to_deobf]
    plt.imshow(stds, cmap='viridis', interpolation='nearest')
    plt.savefig(os.path.join(save_dir, env_name, f'std_version_{vecobs_quantizer}.png'))
    
    # plot number of collisions
    fig = plt.figure()
    plt.bar(np.arange(len(num_collisions)), num_collisions)
    plt.ylim(0, 250)
    plt.ylabel('Collisions')
    plt.xlabel('Action ID')
    plt.savefig(os.path.join(save_dir, env_name, f'collisions_version_{vecobs_quantizer}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLObtainIronPickaxe-v0')
    parser.add_argument('--save_dir', type=str, default='/home/lieberummaas/datadisk/minerl/vecobs_analysis')
    parser.add_argument('--vecobs_quantizer', type=int, default=0)
    
    args = parser.parse_args()
    
    main(**vars(args))