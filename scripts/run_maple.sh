#!/bin/bash

#cd ../..

# custom config
DATA="/home/jianglongyu/mydrive/clip_dataset/"
TRAINER=MaPLe
ABSDIR=/home/jianglongyu/Documents/mllm/fine_tuning_CLIP

DATASET=my_dataset
SEED=1024

CFG=vit_b32_maple
SHOTS=500

DIR=output/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}_1
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file ${ABSDIR}/configs/datasets/${DATASET}.yaml \
--config-file ${ABSDIR}/configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${ABSDIR}/${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES base