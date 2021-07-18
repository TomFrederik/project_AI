import env_wrappers
import models
import datasets
import train_BC

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import argparse
import os
from pyvirtualdisplay import Display
import gym

MAX_TEST_EPISODE_LEN = 15000

def load_model_and_eval(env_name, model_path, test_episodes):
    model = models.BCLinear.load_from_checkpoint(model_path)
    evaluate_BC(env_name, model, test_episodes)

def evaluate_BC(env_name, model, test_episodes):
    display = Display(visible=0, size=(400, 300))
    display.start()

    env = gym.make(env_name)

    num_actions = model.centroids.shape[0]
    action_list = np.arange(num_actions)

    n_frames = 5

    for episode in range(test_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # only sample new action every N frames
            if steps % n_frames == 0:    
                obs_pov = model.VAE.encode_only(torch.from_numpy(obs['pov'].transpose(2, 0, 1)[None].astype(np.float32) / 255).to(model.device))
                obs_vec = torch.from_numpy(obs['vector'][None].astype(np.float32)).to(model.device)
                obs = torch.cat([obs_pov, obs_vec], dim=1)
                
                # Turn logits into probabilities
                probabilities = torch.softmax(model(obs), dim=1)[0]
                # Into numpy
                probabilities = probabilities.detach().cpu().numpy()
                # Sample action according to the probabilities
                discrete_action = np.random.choice(action_list, p=probabilities)

                # Map the discrete action to the corresponding action centroid (vector)
                action = model.centroids[discrete_action].numpy()
                minerl_action = {"vector": action}
            obs, reward, done, info = env.step(minerl_action)
            total_reward += reward
            steps += 1
            if steps >= MAX_TEST_EPISODE_LEN:
                break

        #env.release()
        #env.play()
        print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to encoding model')
    parser.add_argument('--env_name')
    parser.add_argument('--test_episodes', default=10, type=int, help='Number of episodes to evaluate the model on')

    args = vars(parser.parse_args())

    load_model_and_eval(**args)