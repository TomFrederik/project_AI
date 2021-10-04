#!/bin/bash
LOGDIR="/home/lieberummaas/datadisk/minerl/experiment_logs"
VQVAE_PATH="/home/lieberummaas/datadisk/minerl/run_logs/VQVAE/MineRLTreechopVectorObf-v0/lightning_logs/version_0/checkpoints/last.ckpt"
VAE_PATH="/home/lieberummaas/datadisk/minerl/run_logs/VAE/MineRLTreechopVectorObf-v0/lightning_logs/version_6/checkpoints/last.ckpt"
for feature_extractor_cls in vqvae conv vae
    do
        if $feature_extractor_cls = vqvae
        then
            PATH=$VQVAE_PATH
        fi
        if $feature_extractor_cls = vae
        then
            PATH=$VAE_PATH
        fi
        if $feature_extractor_cls = conv
        then
            PATH=""
        fi
        echo "Now training ${feature_extractor_cls}"
        python PretrainDQN.py --log_dir $LOGDIR --feature_extractor_cls $feature_extractor_cls --num_workers 6 --train_feature_extractor --feature_extractor_path $PATH
    done
~         