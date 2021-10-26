#!/bin/bash

VQVAE_PATH="/home/lieberummaas/datadisk/minerl/experiment_logs/VisualModel/2p1s1xsz/checkpoints/last.ckpt"
VAE_PATH="/home/lieberummaas/datadisk/minerl/experiment_logs/VisualModel/1he1xe7g/checkpoints/last.ckpt"
MDN_VQVAE_PATH="/home/lieberummaas/datadisk/minerl/run_logs/DynamicsModel/1coxisei/checkpoints/last.ckpt"
MDN_VAE_PATH="/home/lieberummaas/datadisk/minerl/run_logs/DynamicsModel/iws5evxw/checkpoints/last.ckpt"

python DQfD_pretrain.py --visual_model_cls conv --num_workers 6 --unfreeze_visual_model

python DQfD_pretrain.py --visual_model_cls vqvae --visual_model_path ${VQVAE_PATH} --num_workers 6 --use_one_hot
python DQfD_pretrain.py --visual_model_cls vqvae --visual_model_path ${VQVAE_PATH} --num_workers 6 --use_one_hot --dynamics_model_cls mdn --dynamics_model_path ${MDN_VQVAE_PATH}

python DQfD_pretrain.py --visual_model_cls vae --visual_model_path ${VAE_PATH} --num_workers 6
python DQfD_pretrain.py --visual_model_cls vae --visual_model_path ${VAE_PATH} --num_workers 6 --dynamics_model_cls mdn --dynamics_model_path ${MDN_VAE_PATH}