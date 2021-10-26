#!/bin/bash

VAE="/home/lieberummaas/datadisk/minerl/experiment_logs/VisualModel/1he1xe7g/checkpoints/last.ckpt"
VQVAE="/home/lieberummaas/datadisk/minerl/experiment_logs/VisualModel/2p1s1xsz/checkpoints/last.ckpt"
MDN_VAE="/home/lieberummaas/datadisk/minerl/run_logs/DynamicsModel/iws5evxw/checkpoints/last.ckpt"
MDN_VQVAE="/home/lieberummaas/datadisk/minerl/run_logs/DynamicsModel/1coxisei/checkpoints/last.ckpt"

VAE_1="/home/lieberummaas/datadisk/minerl/run_logs/q_net_4j3fo1ix_1.pt"
VAE_2="/home/lieberummaas/datadisk/minerl/run_logs/q_net_4j3fo1ix_2.pt"
VAE_3="/home/lieberummaas/datadisk/minerl/run_logs/q_net_4j3fo1ix_3.pt"

VQVAE_1="/home/lieberummaas/datadisk/minerl/run_logs/q_net_omkpq18d_1.pt"
VQVAE_2="/home/lieberummaas/datadisk/minerl/run_logs/q_net_omkpq18d_2.pt"
VQVAE_3="/home/lieberummaas/datadisk/minerl/run_logs/q_net_omkpq18d_3.pt"

CONV_1="/home/lieberummaas/datadisk/minerl/run_logs/q_net_169ajjs0_1.pt"
CONV_2="/home/lieberummaas/datadisk/minerl/run_logs/q_net_169ajjs0_2.pt"
CONV_3="/home/lieberummaas/datadisk/minerl/run_logs/q_net_169ajjs0_3.pt"

VAE_MDN_1="/home/lieberummaas/datadisk/minerl/run_logs/q_net_14aujoy5_1.pt"
VAE_MDN_2="/home/lieberummaas/datadisk/minerl/run_logs/q_net_14aujoy5_2.pt"
VAE_MDN_3="/home/lieberummaas/datadisk/minerl/run_logs/q_net_14aujoy5_3.pt"

VQVAE_MDN_1="/home/lieberummaas/datadisk/minerl/run_logs/q_net_3ls5pc7n_1.pt"
VQVAE_MDN_2="/home/lieberummaas/datadisk/minerl/run_logs/q_net_3ls5pc7n_2.pt"
VQVAE_MDN_3="/home/lieberummaas/datadisk/minerl/run_logs/q_net_3ls5pc7n_3.pt"

# xvfb-run -a python eval_trained_dqfd.py --model_path ${VAE_1} --visual_model_cls "vae" --visual_model_path ${VAE}
xvfb-run -a python eval_trained_dqfd.py --model_path ${VAE_2} --visual_model_cls "vae" --visual_model_path ${VAE}
xvfb-run -a python eval_trained_dqfd.py --model_path ${VAE_3} --visual_model_cls "vae" --visual_model_path ${VAE}

xvfb-run -a python eval_trained_dqfd.py --model_path ${VQVAE_1} --visual_model_cls "vqvae" --visual_model_path ${VQVAE}
xvfb-run -a python eval_trained_dqfd.py --model_path ${VQVAE_2} --visual_model_cls "vqvae" --visual_model_path ${VQVAE}
xvfb-run -a python eval_trained_dqfd.py --model_path ${VQVAE_3} --visual_model_cls "vqvae" --visual_model_path ${VQVAE}

xvfb-run -a python eval_trained_dqfd.py --model_path ${CONV_1} --visual_model_cls "conv"
xvfb-run -a python eval_trained_dqfd.py --model_path ${CONV_2} --visual_model_cls "conv"
xvfb-run -a python eval_trained_dqfd.py --model_path ${CONV_3} --visual_model_cls "conv"

xvfb-run -a python eval_trained_dqfd.py --model_path ${VAE_MDN_1} --visual_model_cls "vae" --dynamics_model_cls "mdn" --visual_model_path ${VAE} --dynamics_model_path ${MDN_VAE}
xvfb-run -a python eval_trained_dqfd.py --model_path ${VAE_MDN_2} --visual_model_cls "vae" --dynamics_model_cls "mdn" --visual_model_path ${VAE} --dynamics_model_path ${MDN_VAE}
xvfb-run -a python eval_trained_dqfd.py --model_path ${VAE_MDN_3} --visual_model_cls "vae" --dynamics_model_cls "mdn" --visual_model_path ${VAE} --dynamics_model_path ${MDN_VAE}

xvfb-run -a python eval_trained_dqfd.py --model_path ${VQVAE_MDN_1} --visual_model_cls "vqvae" --dynamics_model_cls "mdn" --visual_model_path ${VQVAE} --dynamics_model_path ${MDN_VQVAE}
xvfb-run -a python eval_trained_dqfd.py --model_path ${VQVAE_MDN_2} --visual_model_cls "vqvae" --dynamics_model_cls "mdn" --visual_model_path ${VQVAE} --dynamics_model_path ${MDN_VQVAE}
xvfb-run -a python eval_trained_dqfd.py --model_path ${VQVAE_MDN_3} --visual_model_cls "vqvae" --dynamics_model_cls "mdn" --visual_model_path ${VQVAE} --dynamics_model_path ${MDN_VQVAE}