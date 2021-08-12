import argparse
import vqvae
import visual_models
import datasets
from minerl import data
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torchvision as tv


STR_TO_MODEL = {
    'vqvae':vqvae.VQVAE
}

def get_tensors(traj, idx, device):
    frame = traj[idx]
    pov = tv.transforms.functional.to_tensor(frame[0]['pov']).to(device)
    vec = torch.from_numpy(frame[0]['vector'].astype(np.float32)).to(device)
    act = torch.from_numpy(frame[1]['vector'].astype(np.float32)).to(device)
    return pov, vec, act

def load_model_and_eval(model_path, model_class, env_name, data_dir, save_path):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load model
    model = STR_TO_MODEL[model_class].load_from_checkpoint(model_path).to(device)

    # load a single trajectory
    env = data.make(env_name, data_dir)
    trajs = env.get_trajectory_names()
    traj = trajs[5]
    traj_iter = env.load_data(traj)

    # pick some frame
    idx = 100
    num_steps = 50

    # make traj to list, take only obs and action
    traj_list = [sarsd[:2] for sarsd in traj_iter]
    
    # get frames to start from
    tensors = [get_tensors(traj_list, i, model.device) for i in range(len(traj_list))]
    pov = torch.stack([t[0] for t in tensors], dim=0)

    batch_size = 50
    batches = len(pov)//batch_size
    if len(pov) % batch_size > 0:
        batches += 1
    rec_pov = []
    for i in range(batches):
        batch = pov[i*batch_size:(i+1)*batch_size]
        batch = model.reconstruct_only(batch)
        rec_pov.append(batch.detach().cpu())
    rec_pov = torch.cat(rec_pov, dim=0)
    # visualize ground truth and reconstructed output
    images = []
    for i in range(len(pov)):
        images.append(np.concatenate(((pov[i][[2,1,0],...].to('cpu').transpose(1,2).numpy()*255).astype(np.uint8), (rec_pov[i][[2,1,0],...].transpose(1,2).numpy()*255).astype(np.uint8)), axis=1).transpose(2,1,0))
     
    size = images[0].transpose(1,0,2).shape[:-1]

    path = os.path.join(save_path,'visual_images')
    os.makedirs(path, exist_ok=True),
    path = os.path.join(path, f'{model_class}.mp4')
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

    for i in range(len(images)):
        out.write(images[i])
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', help='Path to visual model')
    parser.add_argument('--model_class', help='model class', default='vqvae')
    parser.add_argument('--env_name', default='MineRLObtainIronPickaxeVectorObf-v0')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--save_path', help='Path where the output should be saved', default='/home/lieberummaas/datadisk/minerl')

    args = vars(parser.parse_args())

    load_model_and_eval(**args)