#!/bin/bash
cd /home/linux/srcn
SRCN_B=16 torchrun --nproc_per_node=2 train_multi.py > train.log 2>&1
echo "Training finished with exit code $?" >> train.log
