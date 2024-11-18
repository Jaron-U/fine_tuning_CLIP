#!/bin/bash

#cd ../..

# custom config
DATA="/home/jianglongyu/mydrive/clip_dataset/"
TRAINER=CLIP_Adapter
ABSDIR=/home/jianglongyu/Documents/mllm/Dassl.pytorch

DATASET=my_dataset
SEED=42

CFG=vit_b16_ep5_batch4
SHOTS=16


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file ${ABSDIR}/configs/datasets/${DATASET}.yaml \
    --config-file ${ABSDIR}/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${ABSDIR}/${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file ${ABSDIR}/configs/datasets/${DATASET}.yaml \
    --config-file ${ABSDIR}/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${ABSDIR}/${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi