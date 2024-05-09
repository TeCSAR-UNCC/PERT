#!/bin/bash

SOCKET=`shuf -i 2000-65000 -n 1`

if [ -z "$1$2" ]
then
    printf "Please specify the GPU number and configuration file, then run the script like this:\n$0 gpu_number config_file\n"
    exit -1
fi


if [ -z "$1" ]
then
    echo "Please specify a GPU number to start training phase..."
    exit -1
fi


if [ -z "$2" ]
then
    echo "Please specify a config file to start training phase..."
    exit -1
fi

deepspeed --master_port $SOCKET --include localhost:$1 finetune_peit.py --cfg $2
