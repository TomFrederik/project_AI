import argparse
import streamlit as st
import vqvae
import os
import minerl
import numpy as np
import torch
import einops
import matplotlib.pyplot as plt
from time import time

def init(env_name, model_path, data_dir):
    # set device
    if 'device' not in st.session_state:
        st.session_state.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
        st.session_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = vqvae.VQVAE.load_from_checkpoint(model_path).to(st.session_state.device)
    
    # init env
    pipeline = minerl.data.make(env_name, data_dir)
    traj_names = pipeline.get_trajectory_names()

    return model, pipeline, traj_names

def load_traj_data(pipeline, traj_name):
    traj_data = pipeline.load_data(traj_name)
    
    # unpack data
    obs, *_ = zip(*traj_data)
    pov_obs = [item['pov'] for item in obs]
    pov_obs = torch.from_numpy(einops.rearrange(np.array(pov_obs), 'b h w c -> b c h w').astype(np.float32) / 255).to(st.session_state.device)

    return pov_obs


def reconstruct_data(model, pov_obs):
    
    batch_size = 500
    reconstructed_pov = []
    time1 = time()
    for i in range(len(pov_obs)//batch_size + 1):
        reconstructed_pov.append(model.reconstruct_only(pov_obs[batch_size*i:batch_size*(i+1)]))
        print(f'Batch {i+1} took {time()-time1}s')
        time1 = time()

    reconstructed_pov = torch.cat(reconstructed_pov, dim=0)

    return reconstructed_pov

def select_traj(traj_name=None):
    if traj_name == None:
        pov = load_traj_data(st.session_state.pipeline, st.session_state.selected_traj)
    else:
        pov = load_traj_data(st.session_state.pipeline,traj_name)
    rec_pov = reconstruct_data(st.session_state.model, pov)

    st.session_state.all_losses = ((rec_pov - pov) ** 2).mean(dim=[1,2,3]).cpu().numpy()
    st.session_state.pov = (einops.rearrange(pov.cpu().numpy(), 'b c h w -> b h w c') * 255).astype(np.uint8)
    st.session_state.rec_pov = (einops.rearrange(rec_pov.cpu().numpy(), 'b c h w -> b h w c') * 255).astype(np.uint8)


def update_frame():
    st.session_state.frame = st.session_state.slider_frame

def next_frame():
    st.session_state.frame += 1

def prev_frame():
    if st.session_state.frame > 0:
        st.session_state.frame -= 1

def main(
    env_name,
    model_path,
    data_dir
    ):

    if 'init' not in st.session_state:
        st.session_state.model, st.session_state.pipeline, st.session_state.traj_names = init(env_name, model_path, data_dir)
        st.session_state.model.eval()
        select_traj(st.session_state.traj_names[0])
        st.session_state['init'] = True
        
    if 'frame' not in st.session_state:
            st.session_state.frame = 0

    st.sidebar.selectbox('Choose a trajectory:', options=st.session_state.traj_names, index=0, key='selected_traj', on_change=select_traj)

    st.sidebar.button(label='Previous Frame', on_click=prev_frame)
    st.sidebar.button(label='Next Frame', on_click=next_frame)
        
    #st.sidebar.slider('Frame:', 0, len(st.session_state.pov)-1, value=0, key='frame')
    slider_frame = st.sidebar.slider(label='Frame', min_value=0, max_value=len(st.session_state.pov)-1, value=st.session_state.frame, step=1, on_change=update_frame, key='slider_frame')

    col1, col2 = st.columns([10,10])
    with col1:
        st.image(st.session_state.pov[st.session_state.frame], width=400)

        fig = plt.figure()
        plt.plot(st.session_state.all_losses)
        plt.xlabel('Frame')
        plt.ylabel('Loss')
        plt.axvline(st.session_state.frame, color='r', linestyle='--')
        st.pyplot(fig=fig)
    with col2:
        st.image(st.session_state.rec_pov[st.session_state.frame], width=400)

        st.text(f'Loss: {st.session_state.all_losses[st.session_state.frame]:.5f}')
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--model_path', type=str, default="/home/lieberummaas/datadisk/minerl/run_logs/VQVAE/MineRLTreechopVectorObf-v0/lightning_logs/version_14/checkpoints/last.ckpt")
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    
    args = parser.parse_args()
    
    main(**vars(args))