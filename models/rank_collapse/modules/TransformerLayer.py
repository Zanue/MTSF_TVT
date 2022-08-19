import torch
import torch.nn as nn
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class AfterNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.norm(self.fn(x, *args, **kwargs))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., output_attention=False):
        super(Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.output_attention=output_attention
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

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        att = self.attend(dots)
        attn = self.dropout(att)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if not self.output_attention:
            return self.to_out(out)
        else:
            return self.to_out(out), att


class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., output_attention=False):
        super(CrossAttention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.output_attention=output_attention
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.query_projection = nn.Linear(dim, inner_dim, bias = False)
        self.key_projection = nn.Linear(dim, inner_dim, bias = False)
        self.value_projection = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            # nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, queries, keys, values):
        B, N, E = queries.shape
        _, S, _ = keys.shape
        H = self.heads
        q = self.query_projection(queries).reshape(B, H, N, -1)
        k = self.key_projection(keys).reshape(B, H, S, -1)
        v = self.value_projection(values).reshape(B, H, S, -1)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        att = self.attend(dots)
        attn = self.dropout(att)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        if not self.output_attention:
            return self.to_out(out)
        else:
            return self.to_out(out), att


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., afterNorm=False, decomp=None, d_out=None, output_attention=False):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.decomp = decomp
        self.d_out = d_out
        self.output_attention = output_attention

        if afterNorm:
            Norm = AfterNorm
        else:
            Norm = PreNorm
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, output_attention=output_attention)),
                Norm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

        if d_out is not None:
            self.projection = nn.Linear(dim, d_out)
            
    def forward(self, x):
        total_trend = torch.zeros_like(x) if self.decomp else None

        if not self.output_attention:
            for attn, ff in self.layers:
                if self.decomp is not None:
                    trend, x = self.decomp(x) #[B, C, L]
                    total_trend += trend
                x = attn(x) + x
                x = ff(x) + x
        else:
            atts = []
            for attn, ff in self.layers:
                if self.decomp is not None:
                    trend, x = self.decomp(x) #[B, C, L]
                    total_trend += trend
                res_x, att = attn(x)
                x = res_x + x
                x = ff(x) + x

                atts.append(att)

        if self.decomp is not None:
            x = total_trend + x

        if self.d_out is not None:
            x = self.projection(x)
        
        if not self.output_attention:
            return x
        else:
            return x, atts


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., afterNorm=False, decomp=None, d_out=None, \
        dec_self='att', dec_cross='att', output_attention=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.decomp = decomp
        self.d_out = d_out
        self.output_attention = output_attention

        if afterNorm:
            Norm = AfterNorm
        else:
            Norm = PreNorm
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, output_attention=output_attention)) \
                    if dec_self == 'att' else nn.Identity(), 
                Norm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                Norm(dim, CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, output_attention=output_attention)) \
                    if dec_cross == 'att' else nn.Identity(),
                Norm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

        if d_out is not None:
            self.projection = nn.Linear(dim, d_out)
            
    def forward(self, x, keys, values):
        total_trend = torch.zeros_like(x) if self.decomp else None
        if not self.output_attention:
            for self_attn, ff1, cross_attn, ff2 in self.layers:
                if self.decomp is not None:
                    trend, x = self.decomp(x) #[B, C, L]
                    total_trend += trend
                x = self_attn(x) + x
                x = ff1(x) + x

                if self.decomp is not None:
                    trend, x = self.decomp(x) #[B, C, L]
                    total_trend += trend
                x = cross_attn(x, keys, values) + x
                x = ff2(x) + x
        else:
            self_atts, cross_atts = [], []
            for self_attn, ff1, cross_attn, ff2 in self.layers:
                if self.decomp is not None:
                    trend, x = self.decomp(x) #[B, C, L]
                    total_trend += trend
                res_x, self_att = self_attn(x)
                x = x + res_x
                x = ff1(x) + x

                if self.decomp is not None:
                    trend, x = self.decomp(x) #[B, C, L]
                    total_trend += trend
                res_x, cross_att = cross_attn(x, keys, values)
                x = res_x + x
                x = ff2(x) + x

                self_atts.append(self_att)
                cross_atts.append(cross_att)

        if self.decomp is not None:
            x = total_trend + x

        if self.d_out is not None:
            x = self.projection(x)

        if not self.output_attention:
            return x
        else:
            return x, self_atts, cross_atts
