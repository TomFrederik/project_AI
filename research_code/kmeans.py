import random
from sklearn.cluster import KMeans
import numpy as np
import argparse
import os

def compute_kmeans(env_name, data_dir='./numpy_data', n_clusters=200, save_dir='./kmeans_models'):
    data_path = data_dir + '/' + env_name + '_data.npz'
    print(f'Loading data from {data_path}')
    actions = np.load(data_path)['actions']
    print(f'data.shape = {actions.shape}')

    # Run k-means clustering using scikit-learn.
    print(f"Running KMeans on the action vectors with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(actions)
    action_centroids = kmeans.cluster_centers_

    # make sure that save dir exists
    print(f"KMeans done! Saving to {save_dir}/{env_name}_..")
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir + '/' + env_name + '_centroids.npy', action_centroids)
    np.save(save_dir + '/' + env_name + '_kmeans.npy', kmeans)
    return action_centroids, actions, kmeans


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir')
    parser.add_argument('--env_name')
    parser.add_argument('--n_clusters', default=200, type=int)
    parser.add_argument('--save_dir')

    args = vars(parser.parse_args())

    compute_kmeans(**args)