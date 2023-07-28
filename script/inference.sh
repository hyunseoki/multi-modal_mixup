#!/usr/bin/bash



python ./src/inference.py --weight_folder ~/ssd1/02_src/LG_plant_disease/checkpoint/resnext_mixup_w_csv_align_csv \
                          --model resnext50_32x4d \
                          --comments resnext_mixup_w_csv_align_csv

python ./src/inference.py --weight_folder ~/ssd1/02_src/LG_plant_disease/checkpoint/resnext_mixup_w_csv \
                          --model resnext50_32x4d \
                          --comments resnext_mixup_w_csv

python ./src/inference.py --weight_folder ~/ssd1/02_src/LG_plant_disease/checkpoint/mixup \
                          --model resnext50_32x4d \
                          --comments resnext50_32x4d_mixup

python ./src/inference.py --weight_folder ~/ssd1/02_src/LG_plant_disease/checkpoint/resnext50_32x4d_csv_align \
                          --model resnext50_32x4d \
                          --comments resnext50_32x4d_csv_align