from hash_model import PromptHash
import time
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
import torch.nn.functional as F
from optimization import BertAdam
from utils.calc_utils import calc_neighbor, Visualizer
# from utils.calc_utils import calc_map_k_fast as calc_map_k
from utils.calc_utils import calc_map_k
from load_data import generate_dataset
from utils import get_logger
import numpy as np
from utils.image_path import *
from torch.amp import GradScaler, autocast

dataset_root_path = "./dataset"


class TrainerAsym:
    def __init__(self, args):

        self.args = args
        if isinstance(args.use_amp, str):
            self.use_amp = args.use_amp.lower() == 'true'
        else:
            self.use_amp = bool(args.use_amp)

        if self.use_amp:
            self.amp_dtype = torch.float16 if args.amp_dtype == 'fp16' else torch.bfloat16
            self.scaler = GradScaler(enabled=(self.amp_dtype == torch.float16))
            amp_log_dtype = args.amp_dtype
        else:
            self.amp_dtype = torch.float32
            self.scaler = GradScaler(enabled=False)
            amp_log_dtype = "N/A"

        self.dataset_name = args.dataset
        self.accumulation_steps = args.accumulation_steps

        torch.random.manual_seed(seed=self.args.seed)
        np.random.seed(self.args.seed)  # Python的随机性
        os.environ['PYTHONHASHSEED'] = str(self.args.seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
        np.random.seed(self.args.seed)  # numpy的随机性
        torch.manual_seed(self.args.seed)  # torch的CPU随机性，为CPU设置随机种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)  # torch的GPU随机性，为当前GPU设置随机种子
            torch.cuda.manual_seed_all(self.args.seed) # Add for multi-GPU safety

        torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
        torch.backends.cudnn.deterministic = True  # 选择确定性算法
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        os.makedirs(self.args.save_dir, exist_ok=True)
        base_dir = os.path.abspath('.') #先获取文件路径
        test_name = os.path.basename(base_dir)
        vis_env = '{}_{}_{}'.format(test_name, self.args.dataset, "PromptHash")
        try:
            self.vis = Visualizer(vis_env, port=8097, server="http://127.0.0.1")
        except Exception as e:
            print(f"Warning: Could not connect to Visdom server. Visualization will be disabled. Error: {e}")
            self.vis = None

        self._init_writer()

        self.logger.info('Start logging...')
        self.logger.info(f"Automatic Mixed Precision (AMP) training enabled: {self.use_amp}")
        if self.use_amp:
            self.logger.info(f"--> AMP dtype: {amp_log_dtype}")

        if self.args.is_train:
            log_str_args = "\n"
            for para in self.args.__dict__:
                log_str_args += " " * (40 - len(para)) + str(para) + "=" + str(self.args.__dict__[para]) + "\n"
            self.logger.info(log_str_args)
        else:
            self.logger.info(f"pretrained: {self.args.pretrained}")

        self.rank = self.args.rank
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

        self._init_dataset()
        self._init_model()

        self.best_avg_map = 0
        self.best_epoch = 0

        self.logger.info("Train dataset len: {}".format(len(self.train_loader.dataset)))

        self.k_bits_list = list(map(int, self.args.k_bits_list.split(",")))

        self.ibuf = {}
        self.tbuf = {}
        self.bbuf = {}

        self.extend_bits_list = []
        self.extend_bits_list.extend(self.k_bits_list)
        self.extend_bits_list.append(self.args.auxiliary_bit_dim)

        self.logger.info("Initializing buffers on device...")
        for one in self.extend_bits_list:
            self.ibuf[one] = torch.randn(self.args.train_num, one, dtype=self.amp_dtype).to(self.device, non_blocking=True)
            self.tbuf[one] = torch.randn(self.args.train_num, one, dtype=self.amp_dtype).to(self.device, non_blocking=True)
            self.bbuf[one] = torch.sign(self.ibuf[one] + self.tbuf[one])

        self.run()

    def run(self):
        if self.args.is_train:
            self.train()
        else:
            self.test()

    def _init_writer(self):
        self.logger = get_logger(os.path.join(self.args.save_dir, "train.log" if self.args.is_train else "test.log"))
        with open(os.path.join(self.args.save_dir, "description.txt"), 'w') as f:
            f.close()

    def _init_model(self):
        self.logger.info("init model.")
        self.logger.info("Using ViT & GPT2...")

        # 需要解冻的层
        self.layers_to_unfreeze = [
            # 解冻视觉编码器的最后两层
            'visual.transformer.resblocks.10.ln_1.weight', 'visual.transformer.resblocks.10.ln_1.bias',
            'visual.transformer.resblocks.10.attn.in_proj_weight', 'visual.transformer.resblocks.10.attn.in_proj_bias',
            'visual.transformer.resblocks.10.attn.out_proj.weight', 'visual.transformer.resblocks.10.attn.out_proj.bias',
            'visual.transformer.resblocks.10.ln_2.weight', 'visual.transformer.resblocks.10.ln_2.bias',
            'visual.transformer.resblocks.10.mlp.c_fc.weight', 'visual.transformer.resblocks.10.mlp.c_fc.bias',
            'visual.transformer.resblocks.10.mlp.c_proj.weight', 'visual.transformer.resblocks.10.mlp.c_proj.bias',
            'visual.transformer.resblocks.11.ln_1.weight', 'visual.transformer.resblocks.11.ln_1.bias',
            'visual.transformer.resblocks.11.attn.in_proj_weight', 'visual.transformer.resblocks.11.attn.in_proj_bias',
            'visual.transformer.resblocks.11.attn.out_proj.weight', 'visual.transformer.resblocks.11.attn.out_proj.bias',
            'visual.transformer.resblocks.11.ln_2.weight', 'visual.transformer.resblocks.11.ln_2.bias',
            'visual.transformer.resblocks.11.mlp.c_fc.weight', 'visual.transformer.resblocks.11.mlp.c_fc.bias',
            'visual.transformer.resblocks.11.mlp.c_proj.weight', 'visual.transformer.resblocks.11.mlp.c_proj.bias',
            # 解冻文本编码器的最后两层
            'transformer.resblocks.10.ln_1.weight', 'transformer.resblocks.10.ln_1.bias',
            'transformer.resblocks.10.attn.in_proj_weight', 'transformer.resblocks.10.attn.in_proj_bias',
            'transformer.resblocks.10.attn.out_proj.weight', 'transformer.resblocks.10.attn.out_proj.bias',
            'transformer.resblocks.10.ln_2.weight', 'transformer.resblocks.10.ln_2.bias',
            'transformer.resblocks.10.mlp.c_fc.weight', 'transformer.resblocks.10.mlp.c_fc.bias',
            'transformer.resblocks.10.mlp.c_proj.weight', 'transformer.resblocks.10.mlp.c_proj.bias',
            'transformer.resblocks.11.ln_1.weight', 'transformer.resblocks.11.ln_1.bias',
            'transformer.resblocks.11.attn.in_proj_weight', 'transformer.resblocks.11.attn.in_proj_bias',
            'transformer.resblocks.11.attn.out_proj.weight', 'transformer.resblocks.11.attn.out_proj.bias',
            'transformer.resblocks.11.ln_2.weight', 'transformer.resblocks.11.ln_2.bias',
            'transformer.resblocks.11.mlp.c_fc.weight', 'transformer.resblocks.11.mlp.c_fc.bias',
            'transformer.resblocks.11.mlp.c_proj.weight', 'transformer.resblocks.11.mlp.c_proj.bias',
            # 解冻文本提示编码器
            'prompt_transformer.resblocks.0.ln_1.weight', 'prompt_transformer.resblocks.0.ln_1.bias',
            'prompt_transformer.resblocks.0.attn.in_proj_weight', 'prompt_transformer.resblocks.0.attn.in_proj_bias',
            'prompt_transformer.resblocks.0.attn.out_proj.weight', 'prompt_transformer.resblocks.0.attn.out_proj.bias',
            'prompt_transformer.resblocks.0.ln_2.weight', 'prompt_transformer.resblocks.0.ln_2.bias',
            'prompt_transformer.resblocks.0.mlp.c_fc.weight', 'prompt_transformer.resblocks.0.mlp.c_fc.bias',
            'prompt_transformer.resblocks.0.mlp.c_proj.weight', 'prompt_transformer.resblocks.0.mlp.c_proj.bias',
            'prompt_transformer.resblocks.1.ln_1.weight', 'prompt_transformer.resblocks.1.ln_1.bias',
            'prompt_transformer.resblocks.1.attn.in_proj_weight', 'prompt_transformer.resblocks.1.attn.in_proj_bias',
            'prompt_transformer.resblocks.1.attn.out_proj.weight', 'prompt_transformer.resblocks.1.attn.out_proj.bias',
            'prompt_transformer.resblocks.1.ln_2.weight', 'prompt_transformer.resblocks.1.ln_2.bias',
            'prompt_transformer.resblocks.1.mlp.c_fc.weight', 'prompt_transformer.resblocks.1.mlp.c_fc.bias',
            'prompt_transformer.resblocks.1.mlp.c_proj.weight', 'prompt_transformer.resblocks.1.mlp.c_proj.bias'
        ]

        self.model = PromptHash(layers_to_unfreeze=self.layers_to_unfreeze, args=self.args).to(self.device)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info(f"load pretrained model at {self.args.pretrained}")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=self.device)) # Use self.device

        self.model.float()

        fuse_mamba_params = list(self.model.hash.FuseMamba.parameters())
        hash_other_params = [p for n, p in self.model.hash.named_parameters()
                            if 'FuseMamba' not in n]

        # 定义参数组，为每组设置不同的学习率
        param_groups = [
            {'params': [param for name, param in self.model.clip.named_parameters() if name not in self.layers_to_unfreeze], 'lr': self.args.clip_lr},
            {'params': [param for name, param in self.model.clip.named_parameters() if name in self.layers_to_unfreeze], 'lr': self.args.clip_lr * 10},
            {'params': fuse_mamba_params, 'lr': self.args.lr * 0.1},
            {'params': hash_other_params, 'lr': self.args.lr}
        ]

        if not self.train_loader: raise RuntimeError("train_loader not initialized.")
        self.total_steps = len(self.train_loader) // self.args.accumulation_steps * self.args.epochs
        self.logger.info(f"Optimizer total steps (t_total): {self.total_steps}")

        self.optimizer = BertAdam(
            param_groups,
            lr=self.args.lr,
            warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6,
            t_total=self.total_steps,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset...")

        global dataset_root_path
        self.args.index_file = os.path.join(dataset_root_path, self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join(dataset_root_path, self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join(dataset_root_path, self.args.dataset, self.args.label_file)

        train_data, query_data, retrieval_data = generate_dataset(captionFile=self.args.caption_file,
                                                                  indexFile=self.args.index_file,
                                                                  labelFile=self.args.label_file,
                                                                  dataset_name=self.dataset_name,
                                                                  maxWords=self.args.max_words,
                                                                  imageResolution=self.args.resolution,
                                                                  query_num=self.args.query_num,
                                                                  train_num=self.args.train_num,
                                                                  seed=self.args.seed)

        self.train_labels = train_data.get_all_label().float()
        self.query_labels = query_data.get_all_label().float()
        self.retrieval_labels = retrieval_data.get_all_label().float()

        self.args.retrieval_num = len(self.retrieval_labels)
        self.args.num_class = self.query_labels.shape[1]

        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
            prefetch_factor=2
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=2
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=2
        )

    def change_state(self, mode):
        if mode == "train":
            self.model.train()
        elif mode == "valid":
            self.model.eval()

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info("\n\n\n")
        self.logger.info(
            "####################### Train epochs: %d/%d #######################" % (epoch + 1, self.args.epochs))
        epoch_avg_loss_dict = {'all_loss': 0}

        if self.accumulation_steps == 1:
            if self.optimizer: self.optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} Training")
        for i, batch_data in enumerate(progress_bar):
                try:
                    image, text, key_padding_mask, label, prompt_caption, index = batch_data
                except ValueError as e:
                    self.logger.error(f"Error unpacking batch data at iteration {i}: {e}")
                    continue

                with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    image = image.float().to(self.device, non_blocking=True)
                    label = label.float().to(self.device, non_blocking=True)
                    text = text.to(self.device, non_blocking=True)
                    prompt_caption = prompt_caption.to(self.device, non_blocking=True)
                    key_padding_mask = key_padding_mask.to(self.device, non_blocking=True)

                    try:
                        output_dict = self.model(image, text, prompt_caption, key_padding_mask)
                    except Exception as model_e:
                        self.logger.error(f"Error during model forward pass at iteration {i}: {model_e}")
                        continue

                    _B_batch = {}
                    for one in self.extend_bits_list:
                        img_cls_hash = output_dict['img_cls_hash'].get(one)
                        txt_cls_hash = output_dict['txt_cls_hash'].get(one)

                        if img_cls_hash is not None:
                            self.ibuf[one][index] = img_cls_hash.detach()
                        if txt_cls_hash is not None:
                            self.tbuf[one][index] = txt_cls_hash.detach()

                        _B_batch[one] = self.bbuf[one][index]

                    ALL_LOSS_DICT = self.compute_loss(output_dict, label, _B_batch)

                    loss = 0
                    valid_loss_count = 0
                    for key in ALL_LOSS_DICT:
                        value = ALL_LOSS_DICT[key]
                        if value is not None and not torch.isnan(value) and not torch.isinf(value):
                             loss += value
                             valid_loss_count += 1
                             if key in epoch_avg_loss_dict:
                                 epoch_avg_loss_dict[key] += value.item()
                             else:
                                 epoch_avg_loss_dict[key] = value.item()
                        else:
                             self.logger.warning(f"Invalid loss component {key} in batch {i}")

                    if valid_loss_count == 0:
                        self.logger.warning(f"No valid loss components in batch {i}, skipping step.")
                        continue

                    epoch_avg_loss_dict['all_loss'] += loss.item()

                    if torch.isnan(loss) or torch.isinf(loss):
                         self.logger.warning(f"NaN or Inf total loss detected at iteration {i} before scaling: {loss.item()}. Skipping batch.")
                         if self.accumulation_steps > 1 and (i + 1) % self.accumulation_steps != 0: self.optimizer.zero_grad(set_to_none=True)
                         continue

                    loss_normalized = loss / self.accumulation_steps if self.accumulation_steps != 1 else loss

                if self.optimizer:
                    if self.accumulation_steps != 1:
                        self.scaler.scale(loss_normalized).backward()
                        if (i + 1) % self.accumulation_steps == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

        with torch.no_grad():
            for one in self.extend_bits_list:
                self.bbuf[one] = torch.sign(self.ibuf[one] + self.tbuf[one])

        if self.vis and 'key' in locals() and key in ALL_LOSS_DICT:
            num_batches = len(self.train_loader) if len(self.train_loader) > 0 else 1
            for key_plot in ALL_LOSS_DICT:
                 if key_plot in epoch_avg_loss_dict:
                      avg_loss_plot = epoch_avg_loss_dict[key_plot] / num_batches
                      self.vis.plot(f"Loss/{key_plot}", avg_loss_plot)
            avg_total_loss_plot = epoch_avg_loss_dict['all_loss'] / num_batches
            self.vis.plot("Loss/Total", avg_total_loss_plot)

            lr_list_vis = self.optimizer.get_lr() if self.optimizer and hasattr(self.optimizer, 'get_lr') else []
            first_lr_vis = float(lr_list_vis[0]) if lr_list_vis and isinstance(lr_list_vis[0], (float, int, torch.Tensor)) else 0.0
            self.vis.plot("LR/LearningRate", first_lr_vis)
            if self.use_amp and self.amp_dtype == torch.float16:
                self.vis.plot("AMP/GradScaler_Scale", self.scaler.get_scale())


        avg_loss_log = epoch_avg_loss_dict['all_loss'] / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        self.logger.info(f">>>>>> [{epoch+1}/{self.args.epochs}] all loss: {avg_loss_log:.5f}")
        lr_log_list = self.optimizer.get_lr() if self.optimizer and hasattr(self.optimizer, 'get_lr') else []
        self.logger.info(f"lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(lr_log_list)))])}")


    def train(self):
        self.logger.info("Start train...")

        for epoch in range(self.args.epochs):
            time1 = time.time()
            self.train_epoch(epoch)

            time2 = time.time()
            spend_time = int(time2 - time1)
            self.logger.info(
                f"{self.args.dataset}_{self.args.k_bits_list}. Train epoch [{epoch+1}], spend {spend_time // 60} min, {spend_time % 60} sec") # Use epoch+1

            if (epoch + 1) % self.args.valid_freq == 0 or (epoch + 1) == self.args.epochs:
                valid_start_time = time.time()
                self.valid(epoch)
                valid_end_time = time.time()
                valid_duration = int(valid_end_time - valid_start_time)
                self.logger.info(
                    f"{self.args.dataset}_{self.args.k_bits_list}. Valid epoch [{epoch+1}], spend {valid_duration // 60} min, {valid_duration % 60} sec") # Use epoch+1


        self.logger.info(f">>>>>>> FINISHED {self.args.dataset}_{self.args.k_bits_list}. <<<<<<<")
        self.logger.info(f"Best epoch: {self.best_epoch+1}, best avg_mAP: {round(float(self.best_avg_map), 5)}")


    @torch.no_grad()
    def valid(self, epoch):
        self.logger.info("\n")
        self.logger.info(" Valid: %d/%d " % (epoch+1, self.args.epochs))
        self.change_state(mode="valid")

        qi_dict, qt_dict = self.get_code(self.query_loader, self.args.query_num)
        ri_dict, rt_dict = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        _map_epoch = 0
        map_count = 0

        query_labels_gpu = self.query_labels.to(self.device, non_blocking=True)
        retrieval_labels_gpu = self.retrieval_labels.to(self.device, non_blocking=True)

        for one in self.extend_bits_list:
            if one != self.args.auxiliary_bit_dim:
                _k_ = None
                q_i = qi_dict.get(one)
                q_t = qt_dict.get(one)
                r_i = ri_dict.get(one)
                r_t = rt_dict.get(one)

                if q_i is None or q_t is None or r_i is None or r_t is None:
                    self.logger.warning(f"Missing codes for bit {one} during validation.")
                    continue

                try:
                    mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), query_labels_gpu, retrieval_labels_gpu, k=_k_).item()
                    mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), query_labels_gpu, retrieval_labels_gpu, k=_k_).item()
                except Exception as map_e:
                     self.logger.error(f"Error calculating mAP for {one} bits: {map_e}")
                     mAPi2t, mAPt2i = 0.0, 0.0

                if one != self.args.auxiliary_bit_dim:
                    _map_epoch += mAPi2t + mAPt2i
                    map_count += 2
                    if self.vis:
                        self.vis.plot("mAPi2t_{}".format(one), round(mAPi2t, 5))
                        self.vis.plot("mAPt2i_{}".format(one), round(mAPt2i, 5))

                    self.logger.info(f">>>>>> [{epoch+1}/{self.args.epochs}]")
                    self.logger.info(f"{one} bits: MAP(i->t): {round(mAPi2t, 5)}, MAP(t->i): {round(mAPt2i, 5)}")

        avg_map_epoch = _map_epoch / map_count if map_count > 0 else 0.0

        self.logger.info(f">>>>>> [{epoch+1}/{self.args.epochs}]")
        self.logger.info(f"Avg mAP: {round(avg_map_epoch, 5)}")
        if self.vis:
            self.vis.plot("mAP/Average", avg_map_epoch)

        if avg_map_epoch > self.best_avg_map:
            self.best_epoch = epoch
            self.best_avg_map = avg_map_epoch
            self.logger.info("$$$$$$$$$$$$$$$$$$$$ Best avg maps. $$$$$$$$$$$$$$$$$$$$$$$$")
            self.save_model(epoch)
            for one in self.extend_bits_list:
                if one in qi_dict and one in qt_dict and one in ri_dict and one in rt_dict:
                    _file_name = "PromptHash-" + self.args.dataset + "-" + str(one) + ".mat"
                    self.save_mat(qi_dict[one], qt_dict[one], ri_dict[one], rt_dict[one], _file_name)

        self.logger.info(f"Best epoch: {self.best_epoch+1}, best avg_mAP: {round(float(self.best_avg_map), 5)}")

    def hash_loss_group(self, hi, ht, hi_buffer, ht_buffer, label_sim, B, K, weight=1, type='-16-bits'):
        ALL_LOSS = {}
        hyper_cls_intra = self.args.hyper_cls_intra
        ALL_LOSS[f'intra_i_{type}'] = weight * hyper_cls_intra * self.bayesian_loss(hi_buffer, hi, label_sim)
        ALL_LOSS[f'intra_t_{type}'] = weight * hyper_cls_intra * self.bayesian_loss(ht_buffer, ht, label_sim)
        hyper_cls_inter = self.args.hyper_cls_inter
        ALL_LOSS[f'inter_{type}'] = weight * hyper_cls_inter * \
                                                   (self.bayesian_loss(hi_buffer, ht, label_sim) +
                                                    self.bayesian_loss(ht_buffer, hi, label_sim))
        hyper_quan = self.args.hyper_quan
        B = B.to(hi.device)
        ALL_LOSS[f'quantization_{type}'] = weight * hyper_quan * (
                    self.quantization_loss_2(hi, B, K_bits=K) + self.quantization_loss_2(ht, B, K_bits=K))
        return ALL_LOSS

    def compute_loss(self, output_dict, label, B_batch):
        ALL_LOSS = {}
        label_sim = calc_neighbor(self.train_labels.to(self.device), label.to(self.device))

        img_cls_hash = {}
        txt_cls_hash = {}
        for one in self.extend_bits_list:
            img_cls_hash[one] = output_dict['img_cls_hash'][one]
            txt_cls_hash[one] = output_dict['txt_cls_hash'][one]

        weights_list = [1 for _ in self.k_bits_list]
        weights_list.append(self.args.mu)

        for i, one in enumerate(self.extend_bits_list):
            _loss_dict_group = self.hash_loss_group(
                img_cls_hash[one],
                txt_cls_hash[one],
                self.ibuf[one],
                self.tbuf[one],
                label_sim,
                B_batch[one],
                K=one,
                weight=weights_list[i],
                type=f"-{one}-bits"
            )
            ALL_LOSS.update(_loss_dict_group)

        hyper_lambda = self.args.hyper_lambda
        hyper_global = self.args.hyper_global
        hyper_local = self.args.hyper_local
        res_img_cls = output_dict.get('res_img_cls')
        res_txt_eos = output_dict.get('res_txt_cls')
        res_prompt_eos =  output_dict.get('res_prompt_cls')

        if hyper_local > 0 and res_img_cls is not None and res_prompt_eos is not None:
             ALL_LOSS['Local_Prompt_Alignment'] = hyper_lambda * hyper_local * self.local_prompt_alignment_loss(res_img_cls, res_prompt_eos, temperature=self.args.tao_local)
        if hyper_global > 0 and res_img_cls is not None and res_txt_eos is not None:
             ALL_LOSS['Global_Prompt_Alignment'] = hyper_lambda * hyper_global * self.global_prompt_alignment_loss(res_img_cls, res_txt_eos, temperature=self.args.tao_global)


        if self.args.hyper_recon > 0 and self.args.auxiliary_bit_dim in B_batch:
             _recon_target = B_batch[self.args.auxiliary_bit_dim]
             batch_size = _recon_target.shape[0]
             mu = self.args.mu
             hyper_recon = self.args.hyper_recon

             img_cls_hash_recon_dict = output_dict.get('img_cls_hash_recon', {})
             txt_cls_hash_recon_dict = output_dict.get('txt_cls_hash_recon', {})

             for i, one in enumerate(self.k_bits_list):
                  img_recon = img_cls_hash_recon_dict.get(one)
                  txt_recon = txt_cls_hash_recon_dict.get(one)

                  if img_recon is not None:
                       _recon_target = _recon_target.to(img_recon.device)
                       ALL_LOSS[f'recon_i_{one}'] = mu * hyper_recon * (
                            F.mse_loss(img_recon, _recon_target, reduction='sum') / batch_size)
                  if txt_recon is not None:
                       _recon_target = _recon_target.to(txt_recon.device)
                       ALL_LOSS[f'recon_t_{one}'] = mu * hyper_recon * (
                            F.mse_loss(txt_recon, _recon_target, reduction='sum') / batch_size)

        return {k: v for k, v in ALL_LOSS.items() if v is not None}

    @torch.no_grad()
    def get_code(self, data_loader, length: int):
        self.change_state(mode="valid")

        ibuf = {}
        tbuf = {}
        code_dtype = torch.float32
        for one in self.extend_bits_list:
            if one != self.args.auxiliary_bit_dim:
                ibuf[one] = torch.empty(length, one, dtype=code_dtype, device=self.device)
                tbuf[one] = torch.empty(length, one, dtype=code_dtype, device=self.device)

        processed_indices = 0
        for batch_data in tqdm(data_loader, desc="Generating Codes"):
            try:
                image, text, key_padding_mask, label, prompt_caption, index = batch_data
            except ValueError as e:
                self.logger.error(f"Error unpacking batch data during code generation: {e}")
                continue

            image = image.float().to(self.device, non_blocking=True)
            text = text.to(self.device, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.device, non_blocking=True)
            index = index.to(self.device, non_blocking=True)
            prompt_caption = prompt_caption.to(self.device, non_blocking=True)

            output_dict = self.model(image, text, prompt_caption, key_padding_mask)

            for one in self.extend_bits_list:
                if one != self.args.auxiliary_bit_dim:
                    img_hash = output_dict['img_cls_hash'].get(one)
                    txt_hash = output_dict['txt_cls_hash'].get(one)

                    if img_hash is not None:
                        img_bin = torch.sign(img_hash.detach().to(dtype=code_dtype))
                        ibuf[one][index, :] = img_bin
                    if txt_hash is not None:
                        txt_bin = torch.sign(txt_hash.detach().to(dtype=code_dtype))
                        tbuf[one][index, :] = txt_bin

            processed_indices += len(index)

        if processed_indices != length:
             self.logger.warning(f"Processed indices ({processed_indices}) does not match expected length ({length}) in get_code.")

        return ibuf, tbuf

    def save_model(self, epoch):
        save_path_best = os.path.join(self.args.save_dir, "model_best.pth")
        try:
             torch.save(self.model.state_dict(), save_path_best)
        except Exception as e:
             self.logger.error(f"Error saving model: {e}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, filename):
        save_dir = os.path.join(self.args.save_dir, "mat_results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        try:
            result_dict = {
                'q_img': query_img.cpu().detach().numpy(),
                'q_txt': query_txt.cpu().detach().numpy(),
                'r_img': retrieval_img.cpu().detach().numpy(),
                'r_txt': retrieval_txt.cpu().detach().numpy(),
                'q_l': self.query_labels.cpu().detach().numpy(),
                'r_l': self.retrieval_labels.cpu().detach().numpy()
            }
            scio.savemat(save_path, result_dict)
        except Exception as e:
            self.logger.error(f"Error saving .mat file: {e}")

    def local_prompt_alignment_loss(self, out_1, out_2, temperature=0.07):
        f_out_1 = F.softmax(out_1, dim=-1)
        f_out_2 = F.softmax(out_2, dim=-1)
        m = 0.5 * (f_out_1 + f_out_2)
        js_div = 0.5 * (F.kl_div(f_out_1.log(), m, reduction='batchmean') + F.kl_div(f_out_2.log(), m, reduction='batchmean'))
        affinity = 1.0 / (1.0 + js_div)
        temperature = temperature * affinity.item()
        bz = out_1.size(0)
        targets = torch.arange(bz).type_as(out_1).long()
        scores = torch.matmul(out_1, out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        targets = targets.to(scores.device)
        loss0 = F.cross_entropy(scores, targets)
        loss1 = F.cross_entropy(scores1, targets)
        return 0.5 * (loss0 + loss1)

    def global_prompt_alignment_loss(self, out_1, out_2, temperature=0.07):
        bz = out_1.size(0)
        targets = torch.arange(bz).type_as(out_1).long()
        scores = torch.matmul(out_1, out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        targets = targets.to(scores.device)
        loss0 = F.cross_entropy(scores, targets)
        loss1 = F.cross_entropy(scores1, targets)
        return 0.5 * (loss0 + loss1)

    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        s = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)
        label_sim = label_sim.to(s.device) # Ensure device
        b_loss = torch.mean(F.softplus(s) - label_sim * s) # Stable version
        return b_loss

    def quantization_loss_2(self, hash_feature, B, K_bits):
        batch_size = hash_feature.shape[0]
        if batch_size == 0: return torch.tensor(0.0, device=hash_feature.device)
        B = B.to(hash_feature.device)
        return F.mse_loss(hash_feature, B, reduction='sum') / batch_size / K_bits

    @torch.no_grad()
    def test(self):
        if self.args.pretrained == "" or self.args.pretrained == "MODEL_PATH":
            self.logger.error("test step must load a model! please set the --pretrained argument.")
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")

        self.change_state(mode="valid")

        qi_dict, qt_dict = self.get_code(self.query_loader, self.args.query_num)
        ri_dict, rt_dict = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        _map_epoch = 0
        map_count = 0

        query_labels_gpu = self.query_labels.to(self.device, non_blocking=True)
        retrieval_labels_gpu = self.retrieval_labels.to(self.device, non_blocking=True)

        self.logger.info("\n--- Test Set Performance ---")
        for one in self.extend_bits_list:
            if one != self.args.auxiliary_bit_dim:
                _k_ = None
                q_i = qi_dict.get(one)
                q_t = qt_dict.get(one)
                r_i = ri_dict.get(one)
                r_t = rt_dict.get(one)

                if q_i is None or q_t is None or r_i is None or r_t is None:
                     self.logger.warning(f"Missing codes for bit {one} during test.")
                     continue

                try:
                    mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), query_labels_gpu, retrieval_labels_gpu, topk=_k_).item()
                    mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), query_labels_gpu, retrieval_labels_gpu, topk=_k_).item()
                except Exception as map_e:
                     self.logger.error(f"Error calculating test mAP for {one} bits: {map_e}")
                     mAPi2t, mAPt2i = 0.0, 0.0



                if one != self.args.auxiliary_bit_dim:
                     _map_epoch += mAPi2t + mAPt2i
                     map_count += 2
                     self.logger.info(f"{one} bits: MAP(i->t): {round(mAPi2t, 5)}, MAP(t->i): {round(mAPt2i, 5)}")

                _file_name = f"PromptHash-{self.args.dataset}-{one}bit-test.mat"
                self.save_mat(q_i, q_t, r_i, r_t, _file_name)

        if map_count > 0:
            avg_map_test = _map_epoch / map_count
            self.logger.info(f"avg mAP: {round(avg_map_test, 5)}")
        else:
            self.logger.info("No target bits found to calculate average test mAP.")

        self.logger.info(f"Test results (.mat files) saved in: {os.path.join(self.args.save_dir, 'mat_results')}") # Use subdir
        self.logger.info(">>>>>> Save *.mat data! Exit...")
