from action_vqvae import ActionVQVAE
import argparse
import os
import minerl
import torch

def main(
    env_name,
    data_dir,
    log_dir,
    action_quantizer,
    batch_size,
    num_trajs
):
    
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'    

# load model
model_path = os.path.join(log_dir, 'ActionVQVAE', env_name, 'lightning_logs', 'version_'+str(action_quantizer), 'checkpoints', 'last.ckpt')
model: ActionVQVAE = ActionVQVAE.load_from_checkpoint(model_path).to(device)  
model.eval()


# set up data pipelines
obf_env_name = env_name + 'VectorObf-v0'
deobf_env_name = env_name + '-v0'

obf_pipeline = minerl.data.make(obf_env_name)
deobf_pipeline = minerl.data.make(deobf_env_name)
names = pipeline.get_trajectory_names()

#
unique_actions = {}


# iterate over all trajectories
for name in names:
    # load trajectory
    obf_traj = obf_pipeline.load_data(name)
    deobf_traj = deobf_pipeline.load_data(name)

    # get actions
    _, obf_actions, *_ = zip(obf_traj)
    _, deobf_actions, *_ = zip(deobf_traj)
    print(deobf_actions)
    obf_actions = np.array([ac['vector'] for ac in obf_actions])
    obf_actions = torch.from_numpy(obf_actions.astype(np.float32)).to(device)
    print(f'{obf_actions.shape = }')
    
    # compute quantized actions
    quantized_actions, *_ = model.encode_only(obf_actions)
    print(f'{quantized_actions.shape = }')

    raise ValueError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechop')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--action_quantizer', type=int, default=None, help='Version of the ActionVQVAE to use')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_trajs', type=int, default=0)
    
    args = parser.parse_args()
    
    main(**vars(args))