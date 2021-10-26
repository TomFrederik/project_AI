import argparse
import os
import random
from time import time

import cv2
import einops 
import gym
import numpy as np
import torch

from DQfD_models import QNetwork

def main(
    env_name,
    centroids_dir,
    num_centroids,
    num_episodes,
    max_episode_len,
    model_path,
    video_dir,
    epsilon
):

    # load model
    q_net = QNetwork.load_from_checkpoint(model_path)
    q_net.eval()

    # set run id
    video_dir = os.path.join(video_dir, env_name, q_net.visual_model.__class__.__name__, str(int(time())))

    # check that video_dir exists
    os.makedirs(video_dir, exist_ok=True)
    
    # load clusters
    clusters = np.load(os.path.join(centroids_dir, env_name + f"_{num_centroids}_centroids.npy"))

    # init env
    env = gym.make(env_name)

    for i in range(num_episodes):
        print(f'\nStarting episode {i+1}')
        # reset env
        obs = env.reset()
        done = False
        steps = 0
        pov_list = []
        episode_rew = 0

        while not done:
            print(steps, episode_rew)
            # save pov obs for video creation
            pov_list.append(obs['pov'])

            # extract pov and vec from obs and convert to torch tensors
            pov = einops.rearrange(obs['pov'], 'h w c -> 1 c h w').astype(np.float32) / 255
            vec = einops.rearrange(obs['vector'], 'd -> 1 d')
            pov = torch.from_numpy(np.array(pov)).float()
            vec = torch.from_numpy(np.array(vec)).float()

            # compute q_values
            q_values = q_net.forward(pov=pov, vec_obs=vec)[0]

            # select action
            if random.random() < epsilon:
                action = random.randint(0, q_net.hparams.n_actions)
            else:
                action = torch.argmax(q_values).squeeze().item()

            # take action in environment
            obs, rew, done, info = env.step({'vector':clusters[action]})
            
            episode_rew += rew

            # check stopping criterion
            steps += 1
            if steps >= max_episode_len and (not done):
                print(f'Stopping prematurely')
                break
        print(f'Collected reward: {episode_rew}')
        
        print(f'\nFinished episode {i+1} after {steps} steps. Creating video..\n')
        out = cv2.VideoWriter(os.path.join(video_dir, f'{i}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 25, (64,64))

        for i in range(len(pov_list)):
            out.write(pov_list[i][:,:,::-1])
        out.release()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLNavigateDenseVectorObf-v0')
    parser.add_argument('--centroids_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data/')
    parser.add_argument('--num_centroids', type=int, default=150)
    parser.add_argument('--num_episodes', type=int, default=2)
    parser.add_argument('--max_episode_len', type=int, default=2000)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--video_dir', type=str, default='/home/lieberummaas/datadisk/minerl/pretrain_videos')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Epsilon for epsilon-greedy behaviour')

    main(**vars(parser.parse_args()))
