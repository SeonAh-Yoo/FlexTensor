#!/bin/env bash
running() {
    stdbuf --output=0 --error=0 python3 optimize_gemm_ml4sys.py \
        --target cuda \
        --parallel 1 \
        --method nns \
        --N $1  \
        --M $2  \
        --K $3  \
        --device $4 \
        --log gemm_1209_nns_$4.log \
        1>debug_$1_$2_$3_$4_1209.log 2>debug_$1_$2_$3_$4_1209.log
        # --log gemm_1203_$4.log \
}

set -x
tmp=0
# array=(
#     "1024 1024 512 0"
#     "1024 1024 512 3"
#     "1024 1024 512 4"
#     "512 1024 1024 4"
#     "1024 256 1024 5"
#     "512 128 1024 5"
# )

# for a in "${array[@]}"; do
#     running $a
# done

for ((i=5; i<11; i++)); do
    for ((j=5; j<11; j++)); do
        for ((k=5; k<11; k++)); do
            tmp=$((tmp + 1))
            if (( tmp % 8 != $1 )); then
                continue
            fi
            running $((2**i)) $((2**j)) $((2**k)) $1
        done
    done
done