#!/bin/bash

################# bash -x ***.sh #################
lr=0.001
gpu_rank=0
valid_freq=1
epochs=100
res_name="result/Result_PromptHash_NUSWIDE"
recon=0.001
hyper_cls_inter=5.0
hyper_quan=0.1

python main.py --is-train --dataset nuswide --caption-file caption.txt --query-num 2100 --train-num 10500 --lr "$lr" --rank "$gpu_rank" --valid-freq "$valid_freq" --epochs "$epochs" --result-name "$res_name" --hyper-recon "$recon" --hyper-cls-inter "$hyper_cls_inter" --hyper-quan "$hyper_quan"
