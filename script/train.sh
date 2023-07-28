#!/usr/bin/bash


lr=1e-3
batch_size=8
epoch=300

for seed in 222 333 666 777
do
    for model in resnet50 tf_efficientnetv2_s tf_mobilenetv3_small_minimal_100.in1k tv_densenet121
    do
        if [ $model != 'resnet50' ]
        then
            python ./src/main.py --model $model \
                                    --seed $seed \
                                --batch_size $batch_size \
                                --lr $lr \
                                --epochs $epoch \
                                --comments $model/baseline/$model/$seed
        fi 

        python ./src/main_mixup.py --model $model \
                                --seed $seed \
                                   --batch_size $batch_size \
                                   --lr $lr \
                                   --epochs $epoch \
                                   --comments $model/mixup/$model/$seed

        python ./src/main_mixup.py --model $model \
                                --seed $seed \
                                   --batch_size $batch_size \
                                   --lr $lr \
                                   --epochs $epoch \
                                   --mixup_csv True \
                                   --comments $model/mixup-w-csv/$model/$seed

    done
done
