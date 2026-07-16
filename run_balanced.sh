#!/bin/bash
set -euo pipefail
cd /home/linux/srcn_v2_balanced
export PATH="/home/linux/anaconda3/bin:$PATH"
SRCN_B=${SRCN_B:-48} torchrun --nproc_per_node=4 train_multi.py > train.log 2>&1
