#!/usr/bin/env sh

GPUS=0
DATASET=market
SPLIT=id
RATIO=0.1
CUDA_VISIBLE_DEVICES=${GPUS} python tools/train_net.py --num-gpus 0 \
    --config-file ./configs/CMDM/mgn_R50_moco.yml \
    MODEL.BACKBONE.PRETRAIN_PATH "pre_models/ckpt_latest_moco_def.pth" \
    DATASETS.ROOT "../data" INPUT.DO_AUTOAUG False TEST.EVAL_PERIOD 60 \
    DATASETS.KWARGS "data_name:${DATASET}+split_mode:${SPLIT}+split_ratio:${RATIO}" \
    OUTPUT_DIR "logs/last/moco_r50_def_aug/${DATASET}/${SPLIT}_${RATIO}_fast_def"