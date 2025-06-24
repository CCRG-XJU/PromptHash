#!/bin/bash

################# bash -x ***.sh #################
gpu_rank=0
res_name="Result_PromptHash_NUSWIDE_TEST"
MODEL_PATH=""

python main.py  --dataset nuswide --query-num 2100 --train-num 10500 --rank "$gpu_rank" --result-name "$res_name" --pretrained "$MODEL_PATH"
