#!/bin/bash

VAE="/home/lieberummaas/datadisk/minerl/run_logs/DQfD_pretraining/4j3fo1ix/checkpoints/last.ckpt"
VQVAE="/home/lieberummaas/datadisk/minerl/run_logs/DQfD_pretraining/omkpq18d/checkpoints/last.ckpt"
VAE_MDN="/home/lieberummaas/datadisk/minerl/run_logs/DQfD_pretraining/14aujoy5/checkpoints/last.ckpt"
VQVAE_MDN="/home/lieberummaas/datadisk/minerl/run_logs/DQfD_pretraining/3ls5pc7n/checkpoints/last.ckpt"
CONV="/home/lieberummaas/datadisk/minerl/run_logs/DQfD_pretraining/169ajjs0/checkpoints/last.ckpt"

xvfb-run -a python DQfD_train.py --model_path ${VAE} --run_id "4j3fo1ix" --seed 1
xvfb-run -a python DQfD_train.py --model_path ${VAE} --run_id "4j3fo1ix" --seed 2
xvfb-run -a python DQfD_train.py --model_path ${VAE} --run_id "4j3fo1ix" --seed 3

xvfb-run -a python DQfD_train.py --model_path ${VQVAE} --run_id "omkpq18d" --seed 1
xvfb-run -a python DQfD_train.py --model_path ${VQVAE} --run_id "omkpq18d" --seed 2
xvfb-run -a python DQfD_train.py --model_path ${VQVAE} --run_id "omkpq18d" --seed 3

xvfb-run -a python DQfD_train.py --model_path ${CONV} --run_id "169ajjs0" --seed 1
xvfb-run -a python DQfD_train.py --model_path ${CONV} --run_id "169ajjs0" --seed 2
xvfb-run -a python DQfD_train.py --model_path ${CONV} --run_id "169ajjs0" --seed 3

xvfb-run -a python DQfD_train.py --model_path ${VAE_MDN} --run_id "14aujoy5" --seed 1
xvfb-run -a python DQfD_train.py --model_path ${VAE_MDN} --run_id "14aujoy5" --seed 2
xvfb-run -a python DQfD_train.py --model_path ${VAE_MDN} --run_id "14aujoy5" --seed 3

xvfb-run -a python DQfD_train.py --model_path ${VQVAE_MDN} --run_id "3ls5pc7n" --seed 1
xvfb-run -a python DQfD_train.py --model_path ${VQVAE_MDN} --run_id "3ls5pc7n" --seed 2
xvfb-run -a python DQfD_train.py --model_path ${VQVAE_MDN} --run_id "3ls5pc7n" --seed 3
