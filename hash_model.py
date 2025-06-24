from timm.layers import SwiGLU
from torch.nn import TransformerEncoder, TransformerEncoderLayer, init
from torch.nn.functional import normalize

from model import open_clip
from model.modules import *

class VLPromptLearner(nn.Module):
    def __init__(self, clip_model, n_cls, maxWords, device):
        super().__init__()
        self.device = device
        self.maxWords = maxWords
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.tokenizer = open_clip.tokenizer.SimpleTokenizer()
        self.ctx_init = "This is an image containing"
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
        # 初始化context vectors
        ctx_vectors = torch.empty(self.maxWords, self.ctx_dim, device=device)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = self.ctx_init
        self.n_ctx = len(self.ctx_init)
        print(f"Independent V-L design")
        print(f'Initial text context: "{self.prompt_prefix + " " + "{}"}"')
        print(f"Number of context words (tokens) for Language prompting: {self.n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {self.n_ctx}")
    
    def forward(self, prompt_caption):
        prompt_caption = prompt_caption.to(self.device)
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(prompt_caption), -1, -1)
        return ctx + prompt_caption

class FusionTransMamba(nn.Module):
    def __init__(self,  num_layers=1, hidden_size=1024, nhead=4):
        super(FusionTransMamba, self).__init__()
        self.d_model = hidden_size
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=False)
        self.transformer = TransformerEncoder(encoder_layer= encoder_layer, num_layers= num_layers, enable_nested_tensor=False)
        self.sigal_d = int(self.d_model/2)
        self.inproj = nn.Linear(self.sigal_d, self.sigal_d)
        self.outproj = nn.Linear(self.sigal_d, self.sigal_d)
        self.mamba = MambaLayer(dim=self.sigal_d, d_state=16, d_conv=4, expand=2)
        self.grn1 = GRN(dim=self.sigal_d)
        self.grn2 = GRN(dim=self.d_model)

    def forward(self, img_cls, txt_eos):
        short_img_cls = self.inproj(img_cls)
        short_txt_eos = self.inproj(txt_eos)
        mamba_att = self.mamba(torch.concat((img_cls.unsqueeze(1), txt_eos.unsqueeze(1)), dim = 1))
        img_cls, txt_eos = torch.chunk(mamba_att, chunks=2, dim=0)
        img_cls = self.outproj(self.grn1(img_cls).squeeze())
        txt_eos = self.outproj(self.grn1(txt_eos).squeeze())
        img_cls = 0.5 * img_cls + 0.5 * short_img_cls
        txt_eos = 0.5 * txt_eos + 0.5 * short_txt_eos
        res_temp_cls = torch.concat((img_cls, txt_eos), dim = -1)
        res_temp_cls = self.grn2(res_temp_cls)
        encoder_X = self.transformer(res_temp_cls)
        encoder_X_r = encoder_X.reshape( -1,self.d_model)
        encoder_X_r = normalize(encoder_X_r, p =2 ,dim =-1)
        img_cls, txt_eos = encoder_X_r[:,:self.sigal_d], encoder_X_r[:,self.sigal_d:]
        return img_cls, txt_eos

class MLPLayer(nn.Module):
    """
    LND - LND or ND - ND
    """

    def __init__(self, dim_list, dropout=0., activation='relu'):
        super().__init__()

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlp = nn.Sequential()

        for i in range(len(dim_list) - 2):
            _in = dim_list[i]
            _out = dim_list[i + 1]
            
            self.mlp.add_module(f"linear_{i}", nn.Linear(_in, _out))
            if activation == 'swiglu':
                self.mlp.add_module(f"activate_{i}", SwiGLU(in_features=_out))
            else:    
                self.mlp.add_module(f"activate_{i}", self.activation_layer)
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(p=dropout))

        self.mlp.add_module(f"linear_final", nn.Linear(dim_list[-2], dim_list[-1]))

    def forward(self, x):
        return self.mlp(x)


class ResidualMLPs(nn.Module):
    """
    Residual MLPs
    ***D - ***D
    """

    def __init__(self, org_dim, hidden_dim, dropout=0., num_layers=2, activation='relu'):
        super().__init__()
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        elif activation == 'swiglu':
            self.activation_layer = SwiGLU(in_features=hidden_dim)
        else:
            pass

        self.mlps = nn.ModuleList(nn.Sequential(
            nn.Linear(org_dim, hidden_dim),
            self.activation_layer,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, org_dim),
        ) for i in range(num_layers))

        self.lns = nn.ModuleList(nn.LayerNorm(org_dim) for i in range(num_layers))

    def forward(self, x):
        for i in range(self.num_layers):
            x = x + self.mlps[i](self.lns[i](x))
        return x

class HashingEncoder(nn.Module):
    """
    hashing encoder, linear projection & tach.
    """

    def __init__(self, org_dim, k_bits, ):
        super().__init__()
        self.fc = nn.Linear(org_dim, k_bits)
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        return torch.tanh(self.fc(x))

class HashingDecoder(nn.Module):
    """
    hashing decoder, MLP & tach.
    """

    def __init__(self, org_bit_dim, recon_bit_dim):
        super().__init__()
        self.mlp = MLPLayer(dim_list=[org_bit_dim, recon_bit_dim, recon_bit_dim])
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        return torch.tanh(self.mlp(x))

class HashingModel(nn.Module):
    """
    Hashing model
    """

    def __init__(self, args=None):
        super().__init__()

        self.dropout = dropout = args.dropout
        self.activation = activation = args.activation
        self.res_mlp_layers = res_mlp_layers = args.res_mlp_layers
        self.auxiliary_bit_dim = auxiliary_bit_dim = args.auxiliary_bit_dim
        self.transformer_layers = transformer_layers = args.transformer_layers
        self.concept_num = concept_num = args.concept_num

        clip_embed_dim = 512

        self.k_bits_list = list(map(int, args.k_bits_list.split(",")))  # str -> list

        self.extend_bits_list = []
        self.extend_bits_list.extend(self.k_bits_list)
        self.extend_bits_list.append(self.auxiliary_bit_dim)

        # share weight.
        self.resmlp_i = self.resmlp_t = self.resmlp_p = ResidualMLPs(org_dim=clip_embed_dim, hidden_dim=4 * clip_embed_dim, dropout=dropout, num_layers=res_mlp_layers, activation=activation)

        # share weight.
        self.hash_encoders = nn.ModuleList(
            HashingEncoder(org_dim=clip_embed_dim, k_bits=one)
            for one in self.extend_bits_list
        )
        # share weight.
        self.hash_decoders = nn.ModuleList(
            HashingDecoder(one, auxiliary_bit_dim)
            for one in self.k_bits_list
        )

        self.FuseMamba = FusionTransMamba(num_layers=1, hidden_size=clip_embed_dim * 2, nhead=4)


    def forward(self, img_cls, txt_eos, prompt_eos):
        output_dict = {}
        short_img_cls = img_cls
        short_txt_cls = txt_eos
        short_prompt_eos = prompt_eos
        img_cls, txt_eos = self.FuseMamba(img_cls, txt_eos)
        img_cls, prompt_eos = self.FuseMamba(img_cls, prompt_eos)
        img_cls = 0.5 * short_img_cls + 0.5 * img_cls
        txt_eos = 0.5 * short_txt_cls + 0.5 * txt_eos
        prompt_eos = 0.5 * short_prompt_eos + 0.5 * prompt_eos
        res_img_cls = self.resmlp_i(img_cls)
        res_txt_cls = self.resmlp_t(txt_eos)
        res_prompt_cls = self.resmlp_p(prompt_eos)

        output_dict['res_img_cls'] = F.normalize(res_img_cls, dim=-1)
        output_dict['res_txt_cls'] = F.normalize(res_txt_cls, dim=-1)
        output_dict['res_prompt_cls'] = F.normalize(res_prompt_cls, dim=-1)

        img_feature = res_img_cls
        txt_feature = res_txt_cls
        prompt_feature = res_prompt_cls

        output_dict['img_cls_hash'] = {}
        output_dict['txt_cls_hash'] = {}
        output_dict['prompt_cls_hash'] = {}

        output_dict['img_cls_hash_recon'] = {}
        output_dict['txt_cls_hash_recon'] = {}
        output_dict['prompt_cls_hash_recon'] = {}

        for i, one in enumerate(self.extend_bits_list):
            img_cls_hash = self.hash_encoders[i](img_feature)
            txt_cls_hash = self.hash_encoders[i](txt_feature)
            prompt_cls_hash = self.hash_encoders[i](prompt_feature)

            output_dict['img_cls_hash'][one] = img_cls_hash
            output_dict['txt_cls_hash'][one] = txt_cls_hash
            output_dict['prompt_cls_hash'][one] = prompt_cls_hash

            if one != self.auxiliary_bit_dim:
                img_cls_hash_recon = self.hash_decoders[i](img_cls_hash)
                txt_cls_hash_recon = self.hash_decoders[i](txt_cls_hash)
                prompt_cls_hash_recon = self.hash_decoders[i](prompt_cls_hash)
                output_dict['img_cls_hash_recon'][one] = img_cls_hash_recon
                output_dict['txt_cls_hash_recon'][one] = txt_cls_hash_recon
                output_dict['prompt_cls_hash_recon'][one] = prompt_cls_hash_recon

        return output_dict


class PromptHash(nn.Module):
    def __init__(self, layers_to_unfreeze, args=None):
        super(PromptHash, self, ).__init__()
        self.args = args
        if self.args.dataset == "coco":
            self.n_cls = 80
        if self.args.dataset == "flickr25k":
            self.n_cls = 24
        if self.args.dataset == "nuswide":
            self.n_cls = 21

        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='metaclip_fullcc')

        # freeze CLIP
        if self.args.is_freeze_clip:
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
        
        # 解冻特定层
        for name, param in self.clip.named_parameters():
            # print(name)
            if name in layers_to_unfreeze:
                param.requires_grad = True

        # 检查解冻状态
        for name, param in self.clip.named_parameters():
            if param.requires_grad:
                print(f"Layer {name} is unfrozen")

        self.prompt_learner = VLPromptLearner(clip_model=self.clip, n_cls=self.n_cls, device=args.rank, maxWords=args.max_words)

        self.hash = HashingModel(args=args)

    def forward(self, image, text, prompt_caption, key_padding_mask):
        text_prompt = self.prompt_learner(prompt_caption)
        image_prompt = torch.randn(1, 1, 1, 1)
        img_cls = self.clip.encode_image(image, image_prompt)
        txt_eos, prompt_eos = self.clip.encode_text(text, text_prompt)
        output_dict = self.hash(img_cls, txt_eos, prompt_eos)
        return output_dict
