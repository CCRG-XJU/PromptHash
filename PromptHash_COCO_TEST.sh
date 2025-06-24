#!/bin/bash

################# bash -x ***.sh #################
gpu_rank=1
res_name="Result_PromptHash_COCO_TEST"
MODEL_PATH=""

python main.py --dataset coco --query-num 5000 --train-num 10000 --rank "$gpu_rank" --result-name "$res_name" --pretrained "$MODEL_PATH"
