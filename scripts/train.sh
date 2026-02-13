#!/bin/bash

EPOCHS=100
LR=1e-4
BS=32

python experiments/train.py \
--experiment_name base \
--batch_size $BS \
--epochs $EPOCHS \
--lr $LR \

for ALPHA in 0.2 0.4 0.6 0.8 1.0
do
python experiments/train.py \
--experiment_name mixup_only_a${ALPHA} \
--batch_size $BS \
--epochs $EPOCHS \
--lr $LR \
--use_mixup \
--mixup_alpha $ALPHA
done

for ALPHA in 0.2 0.4 0.6 0.8 1.0
do
python experiments/train.py \
--experiment_name mixup_unlabeled_a${ALPHA} \
--batch_size $BS \
--epochs $EPOCHS \
--lr $LR \
--use_mixup \
--use_unlabeled \
--mixup_alpha $ALPHA
done