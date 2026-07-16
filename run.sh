#!/bin/bash
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! 永不删除 checkpoint.pt !!
# !! 脚本自带 resume，崩了从最近存档继续，不丢进度 !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cd /home/linux/srcn_v2
export PATH="/home/linux/anaconda3/bin:$PATH"
SRCN_B=${SRCN_B:-96} torchrun --nproc_per_node=4 train_multi.py > train.log 2>&1
