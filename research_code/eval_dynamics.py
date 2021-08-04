import argparse
import dynamics_models
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
    'rssm':dynamics_models.RSSM,
    'node':dynamics_models.NODEDynamicsModel,
    'mdn':dynamics_models.MDN_RNN
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
    num_steps = 100

    if model_class == 'node':
        # overwrite model timesteps
        model.timesteps = torch.linspace(0,model.seq_len,model.seq_len, device=model.device)

    # make traj to list, take only obs and action
    traj_list = [sarsd[:2] for sarsd in traj_iter]
    
    # get frames to start from
    tensors = [get_tensors(traj_list, i, model.device) for i in range(idx)]
    pov = torch.stack([t[0] for t in tensors], dim=0)
    vec = torch.stack([t[1] for t in tensors], dim=0)
    act = torch.stack([t[2] for t in tensors], dim=0)
    following_acts = torch.stack([get_tensors(traj_list, idx+i+1, model.device)[2] for i in range(num_steps)], dim=0)
    actions = torch.cat([act, following_acts], dim=0)

    following_povs = torch.stack([get_tensors(traj_list, idx+i+1, model.device)[0] for i in range(num_steps)], dim=0)
    following_vecs = torch.stack([get_tensors(traj_list, idx+i+1, model.device)[1] for i in range(num_steps)], dim=0)


    # encode pov
    enc_pov = model.VAE.encode_only(pov)[2]
    rec_pov = model.VAE.decode_only(enc_pov)

    # prepare model input
    states = torch.cat([enc_pov, vec], dim=1)
    
    if model_class in ['rssm', 'mdn']:
        predicted_states = model.predict_recursively(states, actions, horizon=num_steps)
        pred_pov = predicted_states[:, :128]
        pred_vec = predicted_states[:, 128:]
    else:
        raise NotImplementedError    
    
    
    # decode again into images:
    pred_pov = model.VAE.decode_only(pred_pov.squeeze())

    # visualize predicted pov and actual next frames
    images = []
    for i in range(len(pov)):
        images.append(np.concatenate(((pov[i][[2,1,0],...].to('cpu').transpose(1,2).numpy()*255).astype(np.uint8), (rec_pov[i][[2,1,0],...].transpose(1,2).to('cpu').numpy()*255).astype(np.uint8)), axis=1).transpose(2,1,0))
    for i in range(len(following_povs)):
        images.append(np.concatenate(((following_povs[i][[2,1,0],...].to('cpu').transpose(1,2).numpy()*255).astype(np.uint8), (pred_pov[i][[2,1,0],...].transpose(1,2).to('cpu').numpy()*255).astype(np.uint8)), axis=1).transpose(2,1,0))
     
    size = images[0].transpose(1,0,2).shape[:-1]

    out = cv2.VideoWriter(os.path.join(save_path,'dynamics_imgs',f'{model_class}_dynamics.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

    for i in range(len(images)):
        out.write(images[i])
    out.release()
    
    
    # plot difference between predicted and true vector obs
    fig = plt.figure()
    ax = plt.axes()
    ydata = (pred_vec[0] - following_vecs[0]).detach().cpu().numpy()
    ydata.sort()
    plot = ax.plot(ydata)[0]
    
    def animate(i):
        ydata = (pred_vec[i] - following_vecs[i]).detach().cpu().numpy()
        ydata.sort()
        plot.set_ydata(ydata)
        ax.set_ylim(ydata.min(), ydata.max())
    
    anim = FuncAnimation(fig, animate, interval=100, frames=num_steps-1, )
 
    plt.draw()
    anim.save(os.path.join(save_path, 'dynamics_imgs', f'vec_diff_{model_class}.mp4'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', help='Path to dynamics model')
    parser.add_argument('--model_class', help='model class')
    parser.add_argument('--env_name')
    parser.add_argument('--data_dir')
    parser.add_argument('--save_path', help='Path where the output should be saved')

    args = vars(parser.parse_args())

    load_model_and_eval(**args)