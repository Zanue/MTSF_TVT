import torch
import math
import torch.nn as nn
import numpy as np

# standard ffn
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., activation='gelu'):
        super(MLP, self).__init__()
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

def get_MLPLayer(dim, mlp_dim, dropout, activation, *args, **kwargs):
    return MLP(dim, mlp_dim, dropout, activation)


# from Autoformer, correlation layer
class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.output_attention = output_attention

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, x): #[B, C ,L]
        x = x.unsqueeze(1) #[B, H, C ,L]
        # period-based dependencies
        q_fft = torch.fft.rfft(x, dim=-1)
        k_fft = torch.fft.rfft(x, dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        V = self.time_delay_agg_full(x, corr).squeeze(1) #[B, C, L]

        return V
        # if self.output_attention:
        #     return (V.contiguous(), corr)
        # else:
        #     return (V.contiguous(), None)

def get_AutoCorrelationLayer(factor, output_attention, *args, **kwargs):
    return AutoCorrelation(factor=factor, output_attention=output_attention)


# ########## fourier layer #############
class FourierMLP(nn.Module):
    def __init__(self, dim, n_tokens, shared=True):
        super(FourierMLP, self).__init__()
        num_frequency = dim // 2 + 1
        in_len = num_frequency
        out_len = num_frequency
        self.shared = shared
        
        self.scale = 1. / (in_len * out_len)
        
        if shared:
            self.weights = nn.Parameter(
                self.scale * torch.rand(n_tokens, in_len, out_len, dtype=torch.cfloat))
        else:
            self.weights = nn.Parameter(
                self.scale * torch.rand(in_len, out_len, dtype=torch.cfloat))

    def forward(self, x): # [B, C, L]
        x_ft = torch.fft.rfft(x, dim=-1)
        if self.shared:
            out_ft = torch.einsum("bex,exy->bey", x_ft, self.weights)
        else:
            out_ft = torch.einsum("bex,xy->bey", x_ft, self.weights)
        out = torch.fft.irfft(out_ft, n=x.size(-1))
        # print(x.max(), out.max())
        return out

def get_FourierMLP(dim, n_tokens, *args, **kwargs):
    return FourierMLP(dim=dim, n_tokens=n_tokens)


def get_TimeLayer(timelayer, *args, **kwargs):
    if timelayer == 'mlp':
        return get_MLPLayer(*args, **kwargs)
    elif timelayer == 'AutoCorrelation':
        return get_AutoCorrelationLayer(*args, **kwargs)
    elif timelayer == 'fourier':
        return get_FourierMLP(*args, **kwargs)
    elif timelayer == 'none':
        return nn.Identity()
    else:
        raise Exception('Type {} of timelayer is not implemented!!!'.format(timelayer))
