conda activate torch
$lr = 0.001
$gpu_rank = 0
$valid_freq = 1
$epochs = 100
$res_name = "result/Result_PromptHash_Flickr"
$recon = 0.001
$hyper_cls_inter = 5.0
$hyper_quan = 0.1

python main.py --is-train --dataset flickr25k --query-num 2000 --train-num 10000 --lr $lr --rank $gpu_rank --valid-freq $valid_freq --epochs $epochs --result-name $res_name --hyper-recon $recon --hyper-cls-inter $hyper_cls_inter --hyper-quan $hyper_quan
