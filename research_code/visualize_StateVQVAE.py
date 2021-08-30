import argparse
import streamlit as st
import os
import minerl
import numpy as np
import torch
import einops
from state_vqvae import StateVQVAE
import matplotlib.pyplot as plt
from time import time

def init(env_name, model_path, data_dir):
    # set device
    if 'device' not in st.session_state:
        st.session_state.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
        st.session_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = StateVQVAE.load_from_checkpoint(model_path).to(st.session_state.device)
    stat_path = os.path.join(model.hparams.framevqvae[:-9], 'stats.json')
    print(f'{stat_path = }')
    if os.path.exists(stat_path):
        model.find_data_mean_var(load_from=stat_path)
    else:
        raise NotImplementedError(f"stat_path {stat_path} should exist but doesn't")
    model.eval()
    
    # init env
    pipeline = minerl.data.make(env_name, data_dir)
    traj_names = pipeline.get_trajectory_names()

    return model, pipeline, traj_names

def load_traj_data(pipeline, traj_name):
    traj_data = pipeline.load_data(traj_name)
    
    # unpack data
    obs, actions, *_ = zip(*traj_data)
    pov_obs, vec_obs = [item['pov'] for item in obs], [item['vector'] for item in obs]
    actions = [ac['vector'] for ac in actions]
    pov_obs = torch.from_numpy(einops.rearrange(np.array(pov_obs), 'b h w c -> b c h w').astype(np.float32) / 255).to(st.session_state.device)
    vec_obs = torch.from_numpy(np.array(vec_obs).astype(np.float32)).to(st.session_state.device)
    actions = torch.from_numpy(np.array(actions).astype(np.float32)).to(st.session_state.device)

    return pov_obs, vec_obs, actions

@torch.no_grad()
def reconstruct_data(model, pov_obs, vec_obs, actions):
    
    max_seq_len = 200
    predictions, *_ = model(pov_obs[None,:], vec_obs[None,:], actions[None,:], max_seq_len)
    predictions, *_ = model.vqvae.quantizer(predictions[0], proj=False)

    max_batch_size = 200
    reconstructed_pov = []
    for i in range(len(predictions)//max_batch_size + 1):
        reconstructed_pov.append(st.session_state.model.vqvae.decode_only(predictions[max_batch_size*i:max_batch_size*(i+1)]))

    reconstructed_pov = torch.cat(reconstructed_pov, dim=0)

    return reconstructed_pov

def select_traj(traj_name=None):
    if traj_name == None:
        pov, vec_obs, actions = load_traj_data(st.session_state.pipeline, st.session_state.selected_traj)
    else:
        pov, vec_obs, actions = load_traj_data(st.session_state.pipeline,traj_name)
    rec_pov = reconstruct_data(st.session_state.model, pov, vec_obs, actions)
    pov = pov[1:]
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
    log_dir,
    version,
    data_dir
    ):
    model_path = os.path.join(log_dir, env_name, 'lightning_logs', 'version_'+str(version), 'checkpoints', 'last.ckpt')
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

    col1, col2, col3 = st.columns([10, 10, 10])
    with col1:
        st.image(st.session_state.pov[st.session_state.frame-1], width=300)

        fig = plt.figure()
        plt.plot(st.session_state.all_losses)
        plt.xlabel('Frame')
        plt.ylabel('Loss')
        plt.axvline(st.session_state.frame, color='r', linestyle='--')
        st.pyplot(fig=fig)
    with col2:
        st.image(st.session_state.rec_pov[st.session_state.frame], width=300)

        st.text(f'Loss: {st.session_state.all_losses[st.session_state.frame]:.5f}')
    with col3:
        st.image(st.session_state.pov[st.session_state.frame], width=300)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--log_dir', type=str, default="/home/lieberummaas/datadisk/minerl/run_logs/StateVQVAE")
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    
    args = parser.parse_args()
    
    main(**vars(args))