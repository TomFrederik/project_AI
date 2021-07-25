import torch
import planet_models
import argparse

def main(env_name, action_repeat, rssm_path, max_opt_iter, num_act_sequences, planning_horizon, top_k, exploration_noise):
    
    # collect planet kwargs
    planet_kwargs = {
        'rssm_path':rssm_path, 
        'max_opt_iter':max_opt_iter, 
        'num_act_sequences':num_act_sequences, 
        'planning_horizon':planning_horizon, 
        'top_k':top_k
    }
    
    # set up experiment
    experiment = planet_models.PlaNetExperiment(planet_kwargs, env_name, action_repeat, exploration_noise)
    
    # run
    experiment.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--rssm_path', type=str, default='./')
    parser.add_argument('--max_opt_iter', type=int, default=10)
    parser.add_argument('--num_act_sequences', type=int, default=1000)
    parser.add_argument('--planning_horizon', type=int, default=12)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--exploration_noise', type=float, default=0.3, help='Std of the noise applied to the chosen action')
    
    args = vars(parser.parse_args())
    
    main(**args)