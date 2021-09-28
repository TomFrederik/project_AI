#!/bin/bash
LOGDIR="/home/lieberummaas/datadisk/minerl/experiment_logs"
for num_embeddings in 1 2 4 8 16 32 64 128 256
    do
        echo "Now training with ${num_embeddings} embeddings"
        python vqvae.py --log_dir $LOGDIR --num_embeddings $num_embeddings --suffix $(printf $num_embeddings)
    done
~         