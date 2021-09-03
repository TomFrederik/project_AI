from vecobs_vqvae import VecObsVQVAE
import argparse
import os
import minerl
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

type_to_num = {
    'air':0,
    'iron_axe':1,
    'iron_pickaxe':2,
    'none':3,
    'other':4,
    'stone_axe':5,
    'stone_pickaxe':6,
    'wooden_axe':7,
    'wooden_pickaxe':8
}

def unpack_obs(obs):
    out = []
    for key in obs:
        if key == 'pov':
            continue
        elif key == 'equipped_items.mainhand.type':
            out.append(float(type_to_num[obs['equipped_items.mainhand.type']]))
        elif key.startswith('equipped_items'):
            out.append(float(obs[key]))
        else:
            for item in obs[key]:
                out.append(float(obs[key][item]))
    return out

def main(
    env_name,
    data_dir,
    log_dir,
    vecobs_quantizer,
    num_trajs,
    save_dir
):
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    

    # get env names
    obf_env_name = env_name + 'VectorObf-v0'
    deobf_env_name = env_name + '-v0'
    
    # load model
    model_path = os.path.join(log_dir, 'VecObsVQVAE', obf_env_name, 'lightning_logs', 'version_'+str(vecobs_quantizer), 'checkpoints', 'last.ckpt')
    model: VecObsVQVAE = VecObsVQVAE.load_from_checkpoint(model_path).to(device)  
    model.eval()
    
    # set up data pipelines
    obf_pipeline = minerl.data.make(obf_env_name, data_dir)
    if not os.path.exists(os.path.join(data_dir, deobf_env_name)):
        minerl.data.download(data_dir, environment=deobf_env_name)
    deobf_pipeline = minerl.data.make(deobf_env_name, data_dir)
    names = obf_pipeline.get_trajectory_names()

    # init collection of obs
    unique_deobf = None
    all_obf = None

    # iterate over all trajectories
    for i, name in tqdm(enumerate(names)):
        if i >= num_trajs and num_trajs > 0:
            break
        # load trajectory
        obf_traj = obf_pipeline.load_data(name)
        deobf_traj = deobf_pipeline.load_data(name)

        # get obs
        obf_obs, *_ = zip(*obf_traj)
        deobf_obs, *_ = zip(*deobf_traj)

        deobf_obs = np.array(list(map(unpack_obs, deobf_obs)))
        obf_obs = np.array([obs['vector'] for obs in obf_obs])
        # drop duplicates
        df = pd.DataFrame(deobf_obs)
        df = df.drop_duplicates()
        
        deobf_obs = deobf_obs[df.index.to_list()]
        obf_obs = obf_obs[df.index.to_list()] # only quantize obs which have not been dropped
        obf_obs = torch.from_numpy(obf_obs.astype(np.float32)).to(device)
        #print(f'{deobf_obs.shape}')
        #print(f'{obf_obs.shape = }')

        #print(df.index)

        # compute quantized obs
        quantized_obs, *_ = model.encode_only(obf_obs)
        quantized_obs = quantized_obs.to('cpu').numpy()
        #print(f'{quantized_obs.shape = }')

        # merge with collection and drop duplicates
        if unique_deobf is None:
            unique_deobf = pd.DataFrame(deobf_obs)
            all_obf = quantized_obs
        else:
            unique_deobf = pd.concat([unique_deobf, df], ignore_index=True)
            all_obf = np.concatenate([all_obf, quantized_obs], axis=0)
            unique_deobf = unique_deobf.drop_duplicates()
            all_obf = all_obf[unique_deobf.index.to_list()]
        print(f'Total unique obs: {all_obf.shape[0]}')
    
    os.makedirs(os.path.join(save_dir, deobf_env_name), exist_ok=True)
    save_path = os.path.join(save_dir, deobf_env_name, f'obs_version_{vecobs_quantizer}.npz')
    np.savez_compressed(save_path, unique_deobf=unique_deobf.to_numpy(), all_obf=all_obf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLObtainIronPickaxe')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--save_dir', type=str, default='/home/lieberummaas/datadisk/minerl/vecobs_analysis')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--vecobs_quantizer', type=int, default=None, help='Version of the ActionVQVAE to use')
    parser.add_argument('--num_trajs', type=int, default=0)
    
    args = parser.parse_args()
    
    main(**vars(args))