import argparse
import os
import random
from time import time

import einops
import gym
import numpy as np
import torch
from tqdm import tqdm
import wandb

from DQfD_models import ConvFeatureExtractor, QNetwork
from DQfD_utils import CombinedMemory, MemoryDataset, load_expert_demo

def main(
    env_name, 
    max_episode_len, 
    model_path, 
    max_env_steps, 
    centroids_path, 
    training_steps_per_iteration,
    lr, 
    horizon, 
    capacity, 
    discount_factor, 
    action_repeat, 
    epsilon, 
    batch_size, 
    num_expert_episodes, 
    data_dir, 
    save_dir,
    alpha, 
    beta_0, 
    agent_p_offset, 
    expert_p_offset, 
    load_from_statedict
):
    
    torch.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # set save dir
    save_path = os.path.join(save_dir, 'q_net.pt')
    print(f'\nSaving model to {save_path}!')


    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # log time
    start = time()

    # set up model
    if load_from_statedict:
        q_net: QNetwork = torch.load(model_path).to(device)
    else:
        q_net = QNetwork.load_from_checkpoint(model_path).to(device)
    
    # init logger
    config = dict(
        env_name=env_name,
        model_path=model_path,
        visual_model_cls=q_net.hparams.visual_model_cls.__name__,
        dynamics_model_cls=q_net.hparams.dynamics_model_cls.__name__ if q_net.hparams.dynamics_model_cls is not None else None
    )

    wandb.init(project='DQfD_training', config=config)
    
    # set up optimization
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss(reduction='none')
    
    # load centroids
    centroids_path = os.path.join(centroids_path, env_name + '_150_centroids.npy') #TODO make sure that it uses the same centroids as in pretraining
    centroids = np.load(centroids_path)
    
    # init memory
    beta=beta_0
    combined_memory_kwargs = dict(
        agent_memory_capacity=capacity,
        horizon=horizon,
        discount_factor=discount_factor,
        p_offset={'agent':agent_p_offset, 'expert':expert_p_offset},
        alpha=alpha,
        beta=beta
    )
    
    # init the dataset
    dataset = MemoryDataset(env_name, data_dir, num_expert_episodes, centroids, combined_memory_kwargs, dynamics_model=q_net.dynamics_model)
    
    # log total environment interactions
    total_env_steps = 0
    num_episodes = 0

    # create env    
    env = gym.make(env_name)

    time1 = time()
    while total_env_steps < max_env_steps:
        obs_list = []
        action_list = []
        rew_list = []
        td_error_list = []
        predictive_state_list = []

        if q_net.dynamics_model is not None:
            predictive_state_list.append(np.zeros(q_net.dynamics_model.hparams.gru_kwargs['hidden_size']))
        else:
            predictive_state_list.append(np.zeros(10))
        
        num_episodes += 1
        print(f'\nStarting episode {num_episodes}...')

        # re-init env
        done = False
        time1 = time()
        obs = env.reset()
        obs_list.append(obs)
        print(f'Resetting the environment took {time()-time1}s')
        
        steps = 0
        total_reward = 0
        
        # prepare input
        obs_pov = torch.from_numpy(einops.rearrange(obs['pov'], 'h w c -> 1 c h w').astype(np.float32) / 255).to(q_net.device)
        obs_vec = torch.from_numpy(einops.rearrange(obs['vector'], 'd -> 1 d').astype(np.float32)).to(q_net.device)

        # go to eval mode
        q_net.eval()
        
        with torch.no_grad():

            # compute q values
            if q_net.dynamics_model is not None:
                q_values = q_net(obs_pov, obs_vec, torch.from_numpy(predictive_state_list[-1].astype(np.float32))[None].to(q_net.device))[0].squeeze()            
            else:
                q_values = q_net(obs_pov, obs_vec)[0].squeeze()
            
            time0 = time()        
            while not done:    
                
                # select new action
                #time1 = time()
                if steps % action_repeat == 0:
                    if np.random.rand(1)[0] < epsilon:
                        action_ind = np.random.randint(centroids.shape[0])
                        highest_q = q_values[action_ind].cpu().item()
                    else:
                        action_ind = torch.argmax(q_values, dim=0).cpu().item()
                        highest_q = q_values[action_ind].cpu().item()

                    # remap action to centroid
                    action = {'vector': centroids[action_ind]}
                #print(f'Selecting an action took {time()-time1}s')
                
                # env step
                #time1 = time()
                obs, rew, done, _ = env.step(action)
                
                # store transition
                obs_list.append(obs)
                rew_list.append(rew)
                action_list.append(action_ind)
                
                #print(f'Taking a step and storing transition took {time()-time1}s')
                
                # prepare input
                #time1 = time()
                obs_pov = torch.from_numpy(einops.rearrange(obs['pov'], 'h w c -> 1 c h w').astype(np.float32) / 255).to(q_net.device)
                obs_vec = torch.from_numpy(einops.rearrange(obs['vector'], 'd -> 1 d').astype(np.float32)).to(q_net.device)
                #print(f'Preparing input took {time()-time1}s')
                
                # compute q values
                if q_net.dynamics_model is not None:
                    sample, *_ = q_net.dynamics_model.visual_model.encode_only(obs_pov) 
                    gru_input = torch.cat([sample, obs_vec, torch.from_numpy(action['vector'])[None].to(q_net.device)], dim=1)[None].float()
                    hidden_states_seq, _ = q_net.dynamics_model.gru(gru_input)
                    predictive_state_list.append(hidden_states_seq[0,0].detach().cpu().float().numpy())
            
                    # record td_error
                    td_error_list.append(np.abs(rew + discount_factor * q_net(obs_pov, obs_vec, hidden_states_seq[0], target=True)[0].squeeze()[torch.argmax(q_values)].cpu().item() - highest_q))
                    
                else:
                    q_values = q_net(obs_pov, obs_vec)[0].squeeze()
                    predictive_state_list.append(predictive_state_list[-1]) # add another zero array to the list

                    # record td_error
                    td_error_list.append(np.abs(rew + discount_factor * q_net(obs_pov, obs_vec, target=True)[0].squeeze()[torch.argmax(q_values)].cpu().item() - highest_q))
                
                #print(obs_list[-1]['vector'].shape)
                #print(len(action_list))
                #print(predictive_state_list[-1].shape)

                # bookkeeping
                total_reward += rew
                steps += 1
                total_env_steps += 1
                if steps >= max_episode_len or total_env_steps == max_env_steps:
                    break

        print(f'\nEpisode {num_episodes}: Total reward: {total_reward}, Duration: {time()-time0}s')
        wandb.log({'Training/Episode Reward': total_reward})

        # store episode into replay memory
        print('\nAdding episode to memory...')
        dataset.add_episode(obs_list, action_list, np.array(rew_list).astype(np.float32), td_error_list, predictive_state_list, memory_id='agent')
        
        # init dataloader
        sampler = torch.utils.data.WeightedRandomSampler(dataset.weights, training_steps_per_iteration*batch_size, replacement=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=6, pin_memory=True)

        # perform k updates
        print(f'\nPerforming {training_steps_per_iteration} parameter updates...')
        total_loss = 0
        updated_td_errors = {}
        
        # go to train mode
        q_net.train()

        for i, batch in tqdm(enumerate(dataloader)):
            # unpack batch
            state, next_state, n_step_state, predictive_state, next_predictive_state, n_step_predictive_state, action, reward, n_step_reward, batch_idcs, weights, expert_mask = batch

            pov, vec = state
            next_pov, next_vec = next_state
            n_step_pov, n_step_vec = n_step_state

            # prepare tensors
            pov = pov.to(device).float()
            vec = vec.to(device).float()
            next_pov = next_pov.to(device).float()
            next_vec = next_vec.to(device).float()
            n_step_pov = n_step_pov.to(device).float()
            n_step_vec = n_step_vec.to(device).float()
            predictive_state = predictive_state.to(device).float()
            next_predictive_state = next_predictive_state.to(device).float()
            n_step_predictive_state = n_step_predictive_state.to(device).float()
            reward = reward.to(device)
            n_step_reward = n_step_reward.to(device)
            action = action.to(device)
            weights = weights.to(device)
            expert_mask = expert_mask.to(device)
            
            # compute q values and choose actions
            if q_net.dynamics_model is not None:
                q_values = q_net(pov, vec, predictive_state)[0].squeeze()            
                next_target_q_values = q_net(next_pov, next_vec, next_predictive_state, target=True).detach()
                next_q_values = q_net(next_pov, next_vec, next_predictive_state)[0].detach()
                n_step_q_values = q_net(n_step_pov, n_step_vec, n_step_predictive_state, target=True).detach()
            else:
                q_values = q_net(pov, vec)[0].squeeze()
                next_target_q_values = q_net(next_pov, next_vec, target=True).detach()
                next_q_values = q_net(next_pov, next_vec)[0].detach()
                n_step_q_values = q_net(n_step_pov, n_step_vec, target=True).detach()
            
            next_action = torch.argmax(next_q_values, dim=1)
            n_step_action = torch.argmax(n_step_q_values, dim=1)
            
            # compute losses
            idcs = torch.arange(0, len(q_values), dtype=torch.long, requires_grad=False)
            selected_q_values = q_values[idcs, action]
            selected_next_q_values = next_target_q_values[idcs, next_action]
            selected_n_step_q_values = n_step_q_values[idcs, n_step_action]

            td_error = reward + discount_factor * next_q_values[idcs, next_action] - q_values[idcs, action]

            J_DQ = (reward + discount_factor * selected_next_q_values - selected_q_values)**2
            one_step_loss = (J_DQ * weights).mean() # importance sampling scaling
            
            n_step_td_errors = n_step_reward + (discount_factor ** horizon) * selected_n_step_q_values - selected_q_values
            n_step_loss = ((n_step_td_errors ** 2) * weights).mean() # importance sampling scaling

            J_E = (expert_mask * q_net._large_margin_classification_loss(q_values, action)).sum() / expert_mask.sum() # only average over actual expert demos
            loss = one_step_loss + n_step_loss + J_E
            total_loss += loss
            
            # logging
            wandb.log({
                'Training/one_step_loss': one_step_loss,
                'Training/n_step_loss': n_step_loss,
                'Training/classification_loss': J_E,
                'Training/ratio_expert_to_agent': expert_mask.detach().float().mean()
            })
            
            # update td errors
            # update towards n_step td error since that ought to be a more accurate estimate of the 'true' error
            dataset.update_td_errors(batch_idcs, torch.abs(n_step_td_errors))
            
            # backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = total_loss.item() / training_steps_per_iteration
        print(f'\nMean loss = {mean_loss}')
        wandb.log({'Training/Loss': mean_loss})

        cur_dur = time()-start
        print(f'Time elapsed so far: {cur_dur // 60}m {cur_dur % 60:.1f}s')
        print(f'Time per iteration: {cur_dur / num_episodes:.1f}s')
        print('\nUpdating target...')
        q_net._update_target()
        print('\nSaving model')
        torch.save(q_net.state_dict(), save_path)
        print('\nUpdating beta...')
        beta = min(beta + (1-beta_0)/max_env_steps, 1)
        wandb.log({'Training/Beta': beta})
        dataset.update_beta(beta)
        print('\nUpdating dataloader...')
        sampler = torch.utils.data.WeightedRandomSampler(dataset.weights, training_steps_per_iteration*batch_size, replacement=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=6, pin_memory=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLNavigateDenseVectorObf-v0')
    parser.add_argument('--centroids_path', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--save_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--max_episode_len', type=int, default=5000)
    parser.add_argument('--max_env_steps', type=int, default=1000000)
    parser.add_argument('--num_expert_episodes', type=int, default=194)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.4, help='PER exponent')
    parser.add_argument('--beta_0', type=float, default=0.6, help='Initial PER Importance Sampling exponent')
    parser.add_argument('--agent_p_offset', type=float, default=0.001)
    parser.add_argument('--expert_p_offset', type=float, default=1)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--capacity', type=int, default=50000)
    parser.add_argument('--training_steps_per_iteration', type=int, default=200)
    parser.add_argument('--model_path', help='Path to the (pretrained) DQN', required=True)
    parser.add_argument('--load_from_statedict', action='store_true', help='loads model from state dict instead, used when continuing training')
    
    args = parser.parse_args()
    
    main(**vars(args))
