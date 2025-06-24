#!/bin/bash

################# bash -x ***.sh #################
gpu_rank=2
res_name="Result_PromptHash_Flickr_TEST"
MODEL_PATH=""

python main.py --dataset flickr25k --query-num 2000 --train-num 10000 --rank "$gpu_rank" --result-name "$res_name" --pretrained "$MODEL_PATH"
