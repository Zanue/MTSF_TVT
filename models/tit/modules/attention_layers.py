import torch.nn as nn
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # q = q * self.scale
        # k = k * self.scale
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print(dots.max())

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def get_Attention(dim, heads, dim_head, dropout, *args, **kwargs):
    return Attention(dim, heads, dim_head, dropout)


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class FourierAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., activation='softmax'):
        super(FourierAttention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.activation = activation

        num_frequency = dim // 2 + 1
        self.scale = 1. / num_frequency

        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        xq_ft = torch.fft.rfft(q * self.scale, dim=-1)
        xk_ft = torch.fft.rfft(k * self.scale, dim=-1)
        xv_ft = torch.fft.rfft(v, dim=-1)
        
        xq_ft = abs(xq_ft)
        xk_ft = abs(xk_ft)
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft, xk_ft))
        # print(xqk_ft.max())
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(xqk_ft, dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft)
        out = torch.fft.irfft(xqkv_ft, n=x.size(-1))
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def get_FourierAttention(dim, heads, dim_head, dropout, *args, **kwargs):
    return FourierAttention(dim, heads, dim_head, dropout)


def get_AttentionLayer(Attentionlayer, *args, **kwargs):
    if Attentionlayer == 'att':
        return get_Attention(*args, **kwargs)
    elif Attentionlayer == 'fourier':
        return get_FourierAttention(*args, **kwargs)
    elif Attentionlayer == 'none':
        return nn.Identity()
    else:
        raise Exception('Type {} of Attentionlayer is not implemented!!!'.format(Attentionlayer))