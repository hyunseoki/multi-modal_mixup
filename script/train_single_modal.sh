#!/usr/bin/bash

batch_size=8
lr=1e-3
model=resnet50

for seed in 111 222 333 444
do
    python ./src/main.py --model $model \
                        --seed $seed \
                        --batch_size $batch_size \
                        --data_type img \
                        --lr $lr \
                        --epochs 300 \
                        --comments baseline/img_only


    python ./src/main.py --model $model \
                        --seed $seed \
                        --batch_size $batch_size \
                        --data_type csv \
                        --lr $lr \
                        --epochs 300 \
                        --comments baseline/csv_only


    python ./src/main.py --model $model \
                        --seed $seed \
                        --batch_size $batch_size \
                        --data_type all \
                        --lr $lr \
                        --epochs 300 \
                        --comments baseline/multi_modal
done