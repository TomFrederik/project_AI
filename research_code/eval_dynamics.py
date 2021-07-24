import argparse
import models
import datasets
from minerl import data
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torchvision as tv


STR_TO_MODEL = {
    'mdn':models.MDNLSTMDynamicsModel,
    'node':models.NODEDynamicsModel
}

def get_tensors(traj, idx, device):
    frame = traj[idx]
    pov = tv.transforms.functional.to_tensor(frame[0]['pov']).to(device) # is this the same transpose used in ToPILImage??
    vec = torch.from_numpy(frame[0]['vector'].astype(np.float32)).to(device)
    act = torch.from_numpy(frame[1]['vector'].astype(np.float32)).to(device)
    return pov, vec, act

def load_model_and_eval(model_path, model_class, env_name, data_dir, save_path):

    # load model
    model = STR_TO_MODEL[model_class].load_from_checkpoint(model_path)

    # load a single trajectory
    env = data.make(env_name, data_dir)
    trajs = env.get_trajectory_names()
    traj = trajs[0]
    traj_iter = env.load_data(traj)

    # pick some frame
    idx = 100
    num_steps = 50

    # overwrite model seq_len
    model.seq_len = num_steps

    if model_class == 'node':
        # overwrite model timesteps
        model.timesteps = torch.linspace(0,model.seq_len,model.seq_len, device=model.device)

    # make traj to list, take only obs and action
    traj_list = [sarsd[:2] for sarsd in traj_iter]
    
    # get frame to start from
    pov, vec, act = get_tensors(traj_list, idx, model.device)
    following_povs = torch.stack([get_tensors(traj_list, idx+i+1, model.device)[0] for i in range(num_steps)], dim=0)

    # encode pov
    enc_pov = model.VAE.encode_only(pov[None,...])

    # prepare model input
    model_input = torch.cat([enc_pov, vec[None,:], act[None,:]], dim=1)
    
    
    if model_class == 'node':
        # pass through model and only keep pov part
        model_output = model(model_input)[1][1:,:,:pov.shape[0]]

    if model_class == 'mdn':
        model_output = model.predict_recursively(model_input)
    
    '''
    plt.figure()
    img = (model.VAE.decode_only(model_input[:,:-128])*255).squeeze().to('cpu').numpy().transpose(1,2,0).astype(np.uint8)
    plt.imshow(img)
    plt.savefig(os.path.join(save_path,'dynamics_imgs',f'{model_class}_000.png'))
    plt.close()
    '''
    assert torch.all(model_input[0,:-128] == model_output[0,0])


    # decode again into images:
    model_output = model.VAE.decode_only(model_output.squeeze())
    #model_output = model_output.reshape((-1, num_steps, model_output.shape[-1]))

    # visualize predicted pov and actual next frames
    images = [np.concatenate(((pov.to('cpu').transpose(1,2).numpy()*255).astype(np.uint8), (model_output[0].transpose(1,2).to('cpu').numpy()*255).astype(np.uint8)), axis=1).transpose(2,1,0)]
    for i in range(len(following_povs)):
        images.append(np.concatenate(((following_povs[i].to('cpu').transpose(1,2).numpy()*255).astype(np.uint8), (model_output[i+1].transpose(1,2).to('cpu').numpy()*255).astype(np.uint8)), axis=1).transpose(2,1,0))
     
    size = images[0].transpose(1,0,2).shape[:-1]


    out = cv2.VideoWriter(os.path.join(save_path,'dynamics_imgs',f'{model_class}_dynamics.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

    for i in range(len(images)):
        plt.figure()
        plt.imshow(images[i])
        plt.savefig(os.path.join(save_path,'dynamics_imgs',f'{model_class}_{i}.png'))
        plt.close()
        out.write(images[i])
    out.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', help='Path to dynamics model')
    parser.add_argument('--model_class', help='model class')
    parser.add_argument('--env_name')
    parser.add_argument('--data_dir')
    parser.add_argument('--save_path', help='Path where the output should be saved')

    args = vars(parser.parse_args())

    load_model_and_eval(**args)