#!/bin/bash

# DIR="/home/dengpanfu/data"
DIR=r'C:\nezih_data\edu\BOUN\Courses\In process\CMPE597 - Sp.Tp. Deep Learning\Project\code\LUPerson-main'

CUDA_VISIBLE_DEVICES=0 python lup_moco.py \
    --data_path "./data/LUP/lmdbs/lmdb" \
    --info_path "./data/LUP/lmdbs/keys.pkl" \
    --eval_path "./data/" \
    --eval_name "market" \
    -a resnet50 \
    --lr 0.3 \
    --optimizer "SGD" \
    -j 32 \
    --batch-size 256 \
    --dist-url 'tcp://localhost:13701' \
    --T 0.1 \
    --aug_type 'reid' \
    --cos 1 \
    --snap_dir 'snapshots/lup/moco' \
    --log_dir 'logs/lup/moco' \
    --mix 1 \
    --auto_resume 1 \
    --save_freq 20 \
    --print-freq 10 \
    --epochs 200 \
    --mean_type "lup" \
    --eval_freq -1
    