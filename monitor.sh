#!/bin/bash
# 30min monitor — 每30分钟自动检查训练
LOG="/home/linux/srcn_v2/train.log"
while true; do
    n=$(grep -c "Loss:" "$LOG" 2>/dev/null)
    echo "===== $(date '+%H:%M:%S') Batch: $n ====="
    grep "| Loss:" "$LOG" 2>/dev/null | tail -5 | grep -oP 'B\d+.*'
    echo "---"
    grep -c '\!\]\|NaN' "$LOG" 2>/dev/null && echo "NaN above" || echo "NaN: 0"
    echo "=================================="
    sleep 1800
done
