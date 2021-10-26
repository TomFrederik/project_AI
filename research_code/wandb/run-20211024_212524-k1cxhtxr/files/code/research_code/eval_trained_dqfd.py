import argparse
import os
import random
from time import time

import einops 
import gym
import numpy as np
import torch
import wandb

from DQfD_models import QNetwork, ConvFeatureExtractor
from dynamics_models import MDN_RNN
from vae_model import VAE
from vqvae import VQVAE

def main(
    env_name,
    centroids_dir,
    num_centroids,
    num_episodes,
    max_episode_len,
    model_path,
    visual_model_cls,
    visual_model_path,
    dynamics_model_cls,
    dynamics_model_path,
    epsilon
):

    vis_model_cls = {
        'vae':VAE,
        'vqvae':VQVAE,
        'conv':ConvFeatureExtractor
    }[visual_model_cls]

    dyn_model_cls = {
        'none':None,
        'mdn':MDN_RNN
    }[dynamics_model_cls]

    # load model
    model_kwargs = {
        'visual_model_path':visual_model_path,
        'optim_kwargs':{'lr':3e-4},
        'n_actions':1000,
        'target_update_rate':100,
        'margin':0.8,
        'discount_factor':0.99,
        'horizon':50,
        'visual_model_cls':vis_model_cls,
        'freeze_visual_model':True,
        'dynamics_model_cls':dyn_model_cls, 
        'dynamics_model_path':dynamics_model_path, 
        'freeze_dynamics_model':True,
        'use_one_hot':True
    }
    q_net = QNetwork(**model_kwargs)
    q_net.load_state_dict(torch.load(model_path))
    q_net.eval()

    config = dict(
        env_name=env_name,
        model_path=model_path,
        visual_model_cls=q_net.hparams.visual_model_cls.__name__,
        dynamics_model_cls=q_net.hparams.dynamics_model_cls.__name__ if q_net.hparams.dynamics_model_cls is not None else None
    )
    mdn = 'mdn' if q_net.hparams.dynamics_model_cls == MDN_RNN else 'none'
    tags=[q_net.hparams.visual_model_cls.__name__, mdn]
    wandb.init(project='Eval_DQfD_train', config=config, tags=tags)

    # load clusters
    clusters = np.load(os.path.join(centroids_dir, env_name + f"_{num_centroids}_centroids.npy"))

    # init env
    env = gym.make(env_name)

    list_of_episode_rewards = []

    for i in range(num_episodes):
        print(f'\nStarting episode {i+1}')
        # reset env
        obs = env.reset()
        done = False
        steps = 0
        pov_list = []
        episode_rew = 0

        if q_net.dynamics_model is not None:
            predictive_state = torch.from_numpy(np.zeros(q_net.dynamics_model.hparams.gru_kwargs['hidden_size']).astype(np.float32)).to(q_net.device)

        while not done:
            # extract pov and vec from obs and convert to torch tensors
            pov = einops.rearrange(obs['pov'], 'h w c -> 1 c h w').astype(np.float32) / 255
            vec = einops.rearrange(obs['vector'], 'd -> 1 d')
            pov = torch.from_numpy(np.array(pov)).float()
            vec = torch.from_numpy(np.array(vec)).float()

            # compute q values
            if q_net.dynamics_model is not None:
                q_values = q_net(pov, vec, predictive_state[None])[0].squeeze()            
            else:
                q_values = q_net(pov, vec)[0].squeeze()

            # select action
            if random.random() < epsilon:
                action = random.randint(0, q_net.hparams.n_actions-1)
            else:
                action = torch.argmax(q_values).squeeze().item()

            # update predictive state
            if q_net.dynamics_model is not None:
                sample, *_ = q_net.dynamics_model.visual_model.encode_only(pov) 
                if q_net.dynamics_model.hparams.visual_model_cls == 'vqvae':
                    sample = einops.rearrange(sample, 'b c d -> b (c d)')
                gru_input = torch.cat([sample, vec, torch.from_numpy(clusters[action])[None].to(q_net.device)], dim=1)[None].float()
                hidden_states_seq, _ = q_net.dynamics_model.gru(gru_input, predictive_state)[None,None]
                predictive_state = hidden_states_seq[0]

            # take action in environment
            obs, rew, done, info = env.step({'vector':clusters[action]})

            if done:
                rew = 100 # fixing a bug in the minerl navigate dense environment
            
            # log reward
            episode_rew += rew

            # check stopping criterion
            steps += 1
            if steps >= max_episode_len and (not done):
                print(f'Step limit reached. Stopping episode!')
                break
        print(f'\nFinished episode {i+1} after {steps} steps.')
        print(f'Collected reward: {episode_rew}')
        list_of_episode_rewards.append(episode_rew)
        wandb.log({
            'Num Episodes':i,
            'Episode steps':steps,
            'Episode reward':episode_rew
        })

    wandb.log({
        'mean_rew':np.mean(list_of_episode_rewards),
        'std_rew':np.std(list_of_episode_rewards)
    })
    print(f'Reward = {np.mean(list_of_episode_rewards)} +- {np.std(list_of_episode_rewards)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLNavigateDenseVectorObf-v0')
    parser.add_argument('--centroids_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data/')
    parser.add_argument('--visual_model_cls', type=str, default='vae')
    parser.add_argument('--visual_model_path', type=str, default='/home/lieberummaas/datadisk/minerl/data/')
    parser.add_argument('--dynamics_model_cls', type=str, default='none')
    parser.add_argument('--dynamics_model_path', type=str, default='')
    parser.add_argument('--num_centroids', type=int, default=1000)
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--max_episode_len', type=int, default=4000)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--epsilon', type=float, default=0.05, help='Epsilon for epsilon-greedy behaviour')

    main(**vars(parser.parse_args()))
