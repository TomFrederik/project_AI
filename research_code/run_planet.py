import torch
import planet_models
import argparse

def main(env_name, action_repeat, rssm_path, max_opt_iter, num_act_sequences, planning_horizon, top_k, exploration_noise, record, video_dir, max_steps, use_clusters, centroids_path):
    
    # collect planet kwargs
    planet_kwargs = {
        'rssm_path':rssm_path, 
        'max_opt_iter':max_opt_iter, 
        'num_act_sequences':num_act_sequences, 
        'planning_horizon':planning_horizon, 
        'top_k':top_k, 
        'use_clusters':use_clusters, 
        'centroids_path':centroids_path
    }
    
    # set up experiment
    experiment = planet_models.PlaNetExperiment(planet_kwargs, env_name, action_repeat, exploration_noise, record, video_dir, max_steps)
    
    # run
    experiment.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--rssm_path', type=str, default='./')
    parser.add_argument('--record', type=bool, default=False, help='Whether to record a video of the agent')
    parser.add_argument('--video_dir', type=str, default='./videos')
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_opt_iter', type=int, default=10)
    parser.add_argument('--num_act_sequences', type=int, default=1000)
    parser.add_argument('--planning_horizon', type=int, default=50)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--exploration_noise', type=float, default=0, help='Std of the noise applied to the chosen action')
    parser.add_argument('--use_clusters', type=bool, default=False)
    parser.add_argument('--centroids_path', type=str, default='./')
    
    args = vars(parser.parse_args())
    
    main(**args)