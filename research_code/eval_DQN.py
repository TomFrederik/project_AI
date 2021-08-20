import gym
from gym.wrappers import Monitor
from pyvirtualdisplay import Display

import argparse
import os

import torch
import pytorch_lightning as pl

import numpy as np
import einops

import PretrainDQN

def main(env_name, num_runs, centroids_path, model_path, action_repeat, video_dir, epsilon, max_test_episode_len):

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = PretrainDQN.PretrainQNetwork.load_from_checkpoint(model_path).to(device)
    model.eval()

    # load centroids
    centroids_path = os.path.join(centroids_path, env_name + '_centroids.npy')
    centroids = np.load(centroids_path)

    # start virtual display
    display = Display(visible=0, size=(400, 300))
    display.start()

    # create env
    #env = gym.make(env_name)
    env = Monitor(gym.make(env_name), video_dir, force=True)
    
    # mean reward over all runs
    mean_rew = 0

    for n in range(num_runs):
        
        print(f'\nStarting episode no. {n+1}...')
        print(f'Saving video to {video_dir}')
        #env = gym.make(env_name)
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # prepare input
            obs_pov = torch.from_numpy(einops.rearrange(obs['pov'], 'h w c -> 1 c h w').astype(np.float32) / 255).to(model.device)
            obs_vec = torch.from_numpy(einops.rearrange(obs['vector'], 'd -> 1 d').astype(np.float32)).to(model.device)
            
            # compute q values
            q_values = model(obs_pov, obs_vec).squeeze()
            
            if steps % action_repeat == 0:
                # select new action
                if np.random.rand(1)[0] < epsilon:
                    action = np.random.randint(centroids.shape[0])
                else:
                    action = torch.argmax(q_values, dim=0).cpu().item()

                # remap action to centroid
                action = {'vector': centroids[action]}

            # env step
            obs, rew, done, _ = env.step(action)
            
            # bookkeeping
            total_reward += rew
            steps += 1
            if steps >= max_test_episode_len:
                break
        
        env.close()

        print(f'Episode #{n + 1} reward: {total_reward}\t\t episode length: {steps}\n')
        mean_rew += total_reward / num_runs
    
    print(f'Mean reward over all runs: {mean_rew}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--centroids_path', type=str, default='/home/lieberummaas/datadisk/minerl/data/')
    parser.add_argument('--video_dir', type=str, default='/home/lieberummaas/datadisk/minerl/videos')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--max_test_episode_len', type=int, default=3000)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--action_repeat', type=int, default=5)
    
    args = parser.parse_args()

    main(**vars(args))