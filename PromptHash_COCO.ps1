conda activate torch
$lr = 0.001
$gpu_rank = 0
$valid_freq = 1
$epochs = 100
$res_name = "result/Result_PromptHash_COCO"
$recon = 0.001
$hyper_cls_inter = 20.0
$hyper_quan = 1.0

python main.py --is-train --dataset coco --query-num 5000 --train-num 10000 --lr $lr --rank $gpu_rank --valid-freq $valid_freq --epochs $epochs --result-name $res_name --hyper-recon $recon --hyper-cls-inter $hyper_cls_inter --hyper-quan $hyper_quan
