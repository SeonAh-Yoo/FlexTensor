#!/bin/env bash
running() {
    stdbuf --output=0 --error=0 python3 optimize_gemm_ml4sys.py \
        --target cuda \
        --parallel 1 \
        --method q \
        --N $1  \
        --M $2  \
        --K $3  \
        --device $4 \
        --use_model \
        --log 1210/gemm_1210_q_w_m.log \
        1>1210/debug_$1_$2_$3_1210_q_w_m.log 2>1210/debug_$1_$2_$3_1210_q_w_m.log
}

set -x
tmp=0
array=(
    "32 32 32" 
    "64 64 64"
    "128 128 128"
    "256 256 256"
    "512 512 512"
    "1024 1024 1024"
    "128 64 256"
    "256 128 512"
    "512 256 1024"
    "64 256 128"
    "256 64 128"
    "512 1024 64"
    "1024 512 256"
    "1024 128 128"
    "256 256 1024"
    "64 64 1024"
)

for a in "${array[@]}"; do
    running $a 0
done

# for ((i=5; i<11; i++)); do
#     for ((j=5; j<11; j++)); do
#         for ((k=5; k<11; k++)); do
#             tmp=$((tmp + 1))
#             if (( tmp % 8 != $1 )); then
#                 continue
#             fi
#             running $((2**i)) $((2**j)) $((2**k)) $1
#         done
#     done
# done