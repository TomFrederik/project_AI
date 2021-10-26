#!/bin/bash
LOGDIR="/home/lieberummaas/datadisk/minerl/experiment_logs"
NUM_EPOCHS=1

for num_embeddings in 16 32 64 128
    do
        for num_variables in 16 32 64 128
            do
                for embedding_dim in 32 64
                    do
                        echo "Now training with ${num_embeddings} embeddings, ${num_variables} num_variables, ${embedding_dim} embedding_dim"
                        python vqvae.py --num_epochs 1 --log_dir $LOGDIR --num_embeddings $num_embeddings --num_variables $num_variables --embedding_dim $embedding_dim #--suffix $(printf $num_embeddings)
                    done
            done
    done