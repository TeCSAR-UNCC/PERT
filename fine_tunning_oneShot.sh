#!/bin/bash

SOCKET=`shuf -i 2000-65000 -n 1`

if [ -z "$1" ]
then
    echo "Please specify a GPU ID numbers to start training phase..."
    exit -1
fi


deepspeed --master_port $SOCKET --include localhost:$1 oneShot_transformer.py --cfg ./configs/PERT/PEiT_ntu_finetune_oneShot_base_3d.yaml 2>&1 | tee oneShot.log
