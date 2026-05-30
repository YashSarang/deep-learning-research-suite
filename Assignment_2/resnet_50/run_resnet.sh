#!/bin/bash

DATA_DIR="../train_data"
EPOCHS=30
SMALL_EPOCHS=20
BATCH_SIZE=1024
SUBSET_BATCH_SIZE=64
PLOT_PATH="../Figures/resnet_50/"
LR=0.001
SEED=42

export CUDA_VISIBLE_DEVICES=2,3
python main.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --small_epochs $SMALL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --subset_batch_size $SUBSET_BATCH_SIZE \
    --plot_path $PLOT_PATH \
    --lr $LR \
    --seed $SEED \
