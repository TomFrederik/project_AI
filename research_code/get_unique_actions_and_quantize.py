from action_vqvae import ActionVQVAE
import argparse
import os
import minerl
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

equip_map = {
    'air':0,
    'iron_axe':1,
    'iron_pickaxe':2,
    'none':3,
    'stone_axe':4,
    'stone_pickaxe':5,
    'wooden_axe':6,
    'wooden_pickaxe':7
}

craft_map = {
    'crafting_table':0,
    'none':1,
    'planks':2,
    'stick':3,
    'torch':4
}

nearbyCraft_map = {
    'furnace':0,
    'iron_axe':1,
    'iron_pickaxe':2,
    'none':3,
    'stone_axe':4,
    'stone_pickaxe':5,
    'wooden_axe':6,
    'wooden_pickaxe':7
} 

nearbySmelt_map = {
    'coal':0,
    'iron_ingot':1,
    'none':2
}

place_map = {
    'cobblestone':0,
    'crafting_table':1,
    'dirt':2,
    'furnace':3,
    'none':4,
    'stone':5,
    'torch':6
}

def unpack_action(action):
    out = []
    for key in action:
        if key == 'camera':
            out.extend(list(action[key]))
        elif key == 'craft':
            out.append(float(craft_map[action[key]]))
        elif key == 'equip':
            out.append(float(equip_map[action[key]]))
        elif key == 'place':
            out.append(float(place_map[action[key]]))
        elif key == 'nearbyCraft':
            out.append(float(nearbyCraft_map[action[key]]))
        elif key == 'nearbySmelt':
            out.append(float(nearbySmelt_map[action[key]]))
        else:
            out.append(float(action[key]))
    return out



def main(
    env_name,
    data_dir,
    log_dir,
    action_quantizer,
    num_trajs,
    save_dir
):
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    

    # get env names
    obf_env_name = env_name + 'VectorObf-v0'
    deobf_env_name = env_name + '-v0'
    
    # load model
    model_path = os.path.join(log_dir, 'ActionVQVAE', obf_env_name, 'lightning_logs', 'version_'+str(action_quantizer), 'checkpoints', 'last.ckpt')
    model: ActionVQVAE = ActionVQVAE.load_from_checkpoint(model_path).to(device)  
    model.eval()
    
    # set up data pipelines
    obf_pipeline = minerl.data.make(obf_env_name, data_dir)
    if not os.path.exists(os.path.join(data_dir, deobf_env_name)):
        minerl.data.download(data_dir, environment=deobf_env_name)
    deobf_pipeline = minerl.data.make(deobf_env_name, data_dir)
    names = obf_pipeline.get_trajectory_names()

    #
    unique_deobf = None
    all_obf = None


    # iterate over all trajectories
    for i, name in tqdm(enumerate(names)):
        if i >= num_trajs and num_trajs > 0:
            break
        # load trajectory
        obf_traj = obf_pipeline.load_data(name)
        deobf_traj = deobf_pipeline.load_data(name)

        # get actions
        _, obf_actions, *_ = zip(*obf_traj)
        _, deobf_actions, *_ = zip(*deobf_traj)

        deobf_actions = np.array(list(map(unpack_action, deobf_actions)))
        obf_actions = np.array([ac['vector'] for ac in obf_actions])
        # drop duplicates
        df = pd.DataFrame(deobf_actions)
        df = df.drop_duplicates()
        
        deobf_actions = deobf_actions[df.index.to_list()]
        obf_actions = obf_actions[df.index.to_list()] # only quantize actions which have not been dropped
        obf_actions = torch.from_numpy(obf_actions.astype(np.float32)).to(device)
        #print(f'{deobf_actions.shape}')
        #print(f'{obf_actions.shape = }')

        #print(df.index)

        # compute quantized actions
        quantized_actions, *_ = model.encode_only(obf_actions)
        quantized_actions = quantized_actions.to('cpu').numpy()
        #print(f'{quantized_actions.shape = }')

        # merge with collection and drop duplicates
        if unique_deobf is None:
            unique_deobf = pd.DataFrame(deobf_actions)
            all_obf = quantized_actions
        else:
            unique_deobf = pd.concat([unique_deobf, df], ignore_index=True)
            all_obf = np.concatenate([all_obf, quantized_actions], axis=0)
            unique_deobf = unique_deobf.drop_duplicates()
            all_obf = all_obf[unique_deobf.index.to_list()]
        print(f'Total unique actions: {all_obf.shape[0]}')
    
    os.makedirs(os.path.join(save_dir, deobf_env_name), exist_ok=True)
    save_path = os.path.join(save_dir, deobf_env_name, f'actions_version_{action_quantizer}.npz')
    np.savez_compressed(save_path, unique_deobf=unique_deobf.to_numpy(), all_obf=all_obf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechop')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--save_dir', type=str, default='/home/lieberummaas/datadisk/minerl/action_analysis')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--action_quantizer', type=int, default=None, help='Version of the ActionVQVAE to use')
    parser.add_argument('--num_trajs', type=int, default=0)
    
    args = parser.parse_args()
    
    main(**vars(args))