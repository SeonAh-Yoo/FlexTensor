#!/bin/env bash
running() {
    stdbuf --output=0 --error=0 python3 optimize_conv2d_ml4sys.py \
        --target cuda \
        --parallel 1 \
        -f $2 -t $(($2 + 1)) \
        -l conv2d_1203.log \
        --shapes $1 \
        1>debug_$1_$2.log 2>debug_$1_$2.log
    # --test conv2d-config.log
}

shapes=( "google" "squeeze" "res" "vgg-16" "vgg-19" "yolo_b8" "mobile_v2")

declare -A my_dict

my_dict["yolo"]=15
my_dict["google"]=49
my_dict["squeeze"]=22
my_dict["res"]=20

set -x
for key in "${!my_dict[@]}"; do
    for ((i=0; i<${my_dict[$key]}; i++)); do
        running $key $i
    done
done