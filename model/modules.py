import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class GRN1(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = nn.Linear(dim, dim)
        self.nin2 = nn.Linear(dim, dim)
        self.norm2 = GRN1(dim=dim)
        self.act2 = nn.SiLU()
        self.act3 = nn.SiLU()

        self.norm = GRN1(dim=dim)
        self.act = nn.SiLU()
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )
        self.temp = nn.Parameter(torch.randn(1).abs(), requires_grad=True)
        self.weight_gate = nn.Parameter(torch.randn(1).abs(), requires_grad=True)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        B, N, C = x.shape
        x = self.nin(x)
        x = self.norm(x)
        x = self.act(x)
        act_x = x
        assert C == self.dim
        n_tokens = x.shape[1:-1].numel()
        img_dims = x.shape[1:-1]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_flip_l = torch.flip(x_flat, dims=[2])
        x_flip_c = torch.flip(x_flat, dims=[1])
        x_flip_lc = torch.flip(x_flat, dims=[1, 2])
        x_ori = self.mamba(x_flat)
        x_mamba_l = self.mamba(x_flip_l)
        x_mamba_c = self.mamba(x_flip_c)
        x_mamba_lc = self.mamba(x_flip_lc)
        x_ori_l = torch.flip(x_mamba_l, dims=[2])
        x_ori_c = torch.flip(x_mamba_c, dims=[1])
        x_ori_lc = torch.flip(x_mamba_lc, dims=[1, 2])
        x_mamba = (x_ori + x_ori_l + x_ori_c + x_ori_lc) * self.temp

        out = x_mamba.transpose(-1, -2).reshape(B, *img_dims, C)
        cos_sim = F.cosine_similarity(out, act_x, dim=-1)
        weight = torch.sigmoid(cos_sim.mean(0)).unsqueeze(-1)

        out = self.weight_gate * weight * out + (1 - weight) * act_x
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.act2(out)
        return out

if __name__ == '__main__':
    mamba = MambaLayer(dim=512).cuda()
    input = torch.rand(32, 196, 512).cuda()
    output = mamba(input)
    print(input.size())
    print(output.size())