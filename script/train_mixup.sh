#!/usr/bin/bash

batch_size=8
lr=1e-3


for model in mobilenetv3_large_100.miil_in21k mobilenetv3_large_100.miil_in21k_ft_in1k mobilenetv3_large_100.ra_in1k mobilenetv3_rw.rmsp_in1k mobilenetv3_small_050.lamb_in1k mobilenetv3_small_075.lamb_in1k mobilenetv3_small_100.lamb_in1k
do

    python ./src/main.py --model $model \
                        --batch_size $batch_size \
                        --lr $lr \
                        --epochs 300 \
                        --comments multi_modal/$model/$lr/$batch_size/baseline

    python ./src/main_mixup.py --model $model \
                            --batch_size $batch_size \
                            --lr $lr \
                            --epochs 300 \
                            --comments multi_modal/$model/$lr/$batch_size/mixup

    python ./src/main_mixup.py --model $model \
                            --batch_size $batch_size \
                            --lr $lr \
                            --epochs 300 \
                            --mixup_csv True \
                            --comments multi_modal/$model/$lr/$batch_size/mixup-w-csv
done