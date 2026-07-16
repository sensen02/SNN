#!/bin/bash
# Fixed DDP/truncated-BPTT training copy. Keeps checkpoints inside this directory.
set -euo pipefail
cd /home/linux/srcn_v2_fixed
export PATH="/home/linux/anaconda3/bin:$PATH"
SRCN_B=${SRCN_B:-96} torchrun --nproc_per_node=4 train_multi.py > train.log 2>&1
