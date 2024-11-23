#!/bin/bash

#cd ../..

# custom config
DATA="/home/jianglongyu/mydrive/clip_dataset/"
TRAINER=CLIP_Adapter
ABSDIR=/home/jianglongyu/Documents/mllm/fine_tuning_CLIP

DATASET=my_dataset
SEED=42

CFG=vit_b32
SHOTS=500

LOADEP=10

COMMON_DIR=shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}_3
MODEL_DIR=output/${COMMON_DIR}
DIR=output/result/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}_1
echo "Evaluating model"
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file ${ABSDIR}/configs/datasets/${DATASET}.yaml \
--config-file ${ABSDIR}/configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${ABSDIR}/${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES base