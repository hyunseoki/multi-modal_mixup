#!/usr/bin/bash

# for fold_idx in 0 1 2 3 4
# do
#     python ./src/main.py --kfold $fold_idx \
#                          --batch_size 16 \
#                          --comments scratch/fold$fold_idx
# done


for fold_idx in 0
do
    python ./src/main.py --kfold $fold_idx \
                         --rnn_backbone /home/hyunseoki/ssd1/02_src/LG_plant_disease/checkpoint/backbone/rnn_backbone.pth \
                         --batch_size 16 \
                         --comments baseline/fold$fold_idx
done
