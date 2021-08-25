from PretrainDQN import PretrainQNetwork
import argparse
import minerl
import os
import torch
import numpy as np
import einops
import streamlit as st
import matplotlib.pyplot as plt

def load_traj(env_name, data_dir, model_version, centroids_path, log_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pipeline = minerl.data.make(env_name, data_dir)
    traj_name = pipeline.get_trajectory_names()[0]
    if 'traj_name' not in st.session_state:
        st.session_state.traj_name = traj_name
    traj_name = st.session_state.traj_name
    traj_data = pipeline.load_data(traj_name)

    # load centroids
    centroids_path = os.path.join(centroids_path, env_name + '_centroids.npy')
    print(f'\nLoading centroids from {centroids_path}...')
    centroids = np.load(centroids_path)

    # unpack data
    obs, actions, rewards, *_ = zip(*traj_data)
    pov_obs, vec_obs = [item['pov'] for item in obs], [item['vector'] for item in obs]
    pov_obs = torch.from_numpy(einops.rearrange(np.array(pov_obs), 'b h w c -> b c h w').astype(np.float32) / 255).to(device)
    vec_obs = torch.from_numpy(np.array(vec_obs).astype(np.float32)).to(device)
    #rewards = torch.from_numpy(np.array(rewards).astype(np.float32)).to(device)
    actions = np.array([ac['vector'] for ac in actions]).astype(np.float32)

    # compute actions
    actions = torch.from_numpy(np.argmin(((centroids[None,:,:] - actions[:,None,:]) ** 2).sum(axis=-1), axis=1).astype(np.int64)).to(device)

    # load Q model
    model = PretrainQNetwork.load_from_checkpoint(os.path.join(log_dir, env_name, 'lightning_logs', 'version_'+str(model_version), 'checkpoints', 'last.ckpt')).to(device)
    model.eval()
    
    # compute q_values
    batch_size = 500
    q_values = []
    for i in range(len(pov_obs) // batch_size + 1):
        with torch.no_grad():
            q_values.append(model(pov_obs[i*batch_size:(i+1)*batch_size], vec_obs[i*batch_size:(i+1)*batch_size])[0])
    q_values = torch.cat(q_values, dim=0).cpu().numpy()

    # compute true q_value for the chosen action
    gamma = model.hparams.gamma
    true_returns = []
    for f in range(len(actions)):
        discount_array = np.array([gamma ** i for i in range(len(actions)-f)])
        true_returns.append(np.sum(discount_array * rewards[f:]))

    # make them numpy again
    pov_obs = (einops.rearrange(pov_obs.cpu().numpy(), 'b c h w -> b h w c') * 255).astype(np.uint8)
    vec_ovs = vec_obs.cpu().numpy()

    return pov_obs, actions, q_values, true_returns

def main(
        env_name,
        model_version,
        data_dir,
        log_dir,
        centroids_path
    ):


    if 'loaded' not in st.session_state:
        st.session_state.pov_obs, st.session_state.actions, st.session_state.q_values, st.session_state.true_returns = load_traj(env_name, data_dir, model_version, centroids_path, log_dir)
        st.session_state.loaded = True
    
    if st.session_state.loaded:
        pov_obs = st.session_state.pov_obs
        # interactive setup
        st.title(f'PretrainedQNetwork - {env_name} - version {model_version}')
        
        if 'frame' not in st.session_state:
            st.session_state.frame = 0
        
        st.write(st.session_state.frame)
        
        st.sidebar.button(label='Next Frame', on_click=next_frame)
        
        slider_frame = st.sidebar.slider(label='Frame', min_value=0, max_value=len(pov_obs)-1, value=st.session_state.frame, step=1, on_change=update_frame, key='slider_frame')

        col1, col2 = st.columns([15,5])
        with col2:
            st.image(pov_obs[st.session_state.frame], width=400, use_column_width=False)
    
        with col1:
            human_action = st.session_state.actions[st.session_state.frame].item()
            human_q = st.session_state.q_values[st.session_state.frame, human_action]
            model_action = np.argmax(st.session_state.q_values[st.session_state.frame])
            model_q = st.session_state.q_values[st.session_state.frame, model_action]
            
            if model_action == human_action:
                reduced_data = np.delete(st.session_state.q_values[st.session_state.frame], [human_action])
            else:
                reduced_data = np.delete(st.session_state.q_values[st.session_state.frame], [human_action, model_action])
            fig = plt.figure()
            plt.bar(np.arange(len(reduced_data)), reduced_data, color='b', label='Q', width=1)
            plt.bar([human_action], [human_q], color='g', label='Human', width=1)
            if model_action != human_action:
                plt.bar([model_action], [model_q], color='r', label='Model', width=1)

            #st.bar_chart(st.session_state.q_values[st.session_state.frame])
            st.pyplot(fig)
            st.text(f'Human Action: {human_action}')
            st.text(f'Model Action: {model_action}')
            
            st.text(f'True return: {st.session_state.true_returns[st.session_state.frame]:.2f}')


    else:
        raise NotImplementedError('Something went wrong!')

def update_frame():
    st.session_state.frame = st.session_state.slider_frame

def next_frame():
    st.session_state.frame += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--model_version', type=int, default=0)
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs/PretrainQNetwork')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--centroids_path', default='/home/lieberummaas/datadisk/minerl/data')

    args = parser.parse_args()

    main(**vars(args))

