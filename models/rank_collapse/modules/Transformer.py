import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=25, permute=False):
        super(series_decomp, self).__init__()
        self.permute = permute
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        if self.permute:
            x = x.permute(0,2,1)
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        if self.permute:
            res = res.permute(0,2,1)
            moving_mean = moving_mean.permute(0,2,1)
        return res, moving_mean




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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., segment=1, tokenization='none'):
        super(Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        if segment > 1:
            assert(dim % segment == 0)
            assert(dim_head % segment == 0)

        self.segment = segment
        self.tokenization = tokenization
        self.heads = heads
        self.scale = (dim_head // segment) ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim // segment, inner_dim * 3 // segment, bias = False)
        if segment > 1:
            self.ffn = nn.Linear(inner_dim // segment, inner_dim // segment, bias = True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim // segment, dim // segment),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        if self.segment > 1:
            batch_size = x.shape[0]
            if self.tokenization == 'time': #[B, L, C]
                x = rearrange(x, 'b (s e) c -> (b c) s e', s=self.segment)
            elif self.tokenization == 'channel': #[B, C, L]
                x = rearrange(x, 'b c (s e) -> (b s) c e', s=self.segment)
            else:
                raise Exception('Type {} of tokenization is error!!!'.format(self.tokenization))

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        if self.segment > 1:
            out = self.ffn(out)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if self.segment > 1:
            if self.tokenization == 'time': #[B, L, C]
                out = rearrange(out, '(b c) s e -> b (s e) c', b=batch_size)
            elif self.tokenization == 'channel': #[B, C, L]
                out = rearrange(out, '(b s) c e -> b c (s e)', b=batch_size)
            else:
                raise Exception('Type {} of tokenization is error!!!'.format(self.tokenization))
        
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., segment=1, value_segment=1, tokenization='none'):
        super(CrossAttention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        if segment > 1:
            assert(dim % segment == 0)
            assert(dim_head % segment == 0)

        self.segment = segment
        self.value_segment = value_segment
        self.tokenization = tokenization
        self.heads = heads
        self.scale = (dim_head // segment) ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.query_projection = nn.Linear(dim // segment, inner_dim // segment, bias = False)
        self.key_projection = nn.Linear(dim // segment, inner_dim // segment, bias = False)
        self.value_projection = nn.Linear(dim // segment, inner_dim // segment, bias = False)
        if segment > 1:
            self.ffn = nn.Linear(inner_dim // segment, inner_dim // segment, bias = True)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim // segment, dim // segment),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, queries, keys, values):

        if self.segment > 1:
            batch_size = queries.shape[0]
            if self.tokenization == 'time': #[B, L, C]
                queries = rearrange(queries, 'b (s e) c -> (b c) s e', s=self.segment)
                keys = rearrange(keys, 'b (s e) c -> (b c) s e', s=self.value_segment)
                values = rearrange(values, 'b (s e) c -> (b c) s e', s=self.value_segment)
            elif self.tokenization == 'channel': #[B, C, L]
                queries = rearrange(queries, 'b c (s e) -> (b s) c e', s=self.segment)
                keys = rearrange(keys, 'b c (s e) -> (b s) c e', s=self.segment)
                values = rearrange(values, 'b c (s e) -> (b s) c e', s=self.segment)
            else:
                raise Exception('Type {} of tokenization is error!!!'.format(self.tokenization))

        B, N, E = queries.shape
        _, S, _ = keys.shape
        H = self.heads
        q = self.query_projection(queries).reshape(B, H, N, -1)
        k = self.key_projection(keys).reshape(B, H, S, -1)
        v = self.value_projection(values).reshape(B, H, S, -1)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        if self.segment > 1:
            out = self.ffn(out)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if self.segment > 1:
            if self.tokenization == 'time': #[B, L, C]
                out = rearrange(out, '(b c) s e -> b (s e) c', b=batch_size)
            elif self.tokenization == 'channel': #[B, C, L]
                out = rearrange(out, '(b s) c e -> b c (s e)', b=batch_size)
            else:
                raise Exception('Type {} of tokenization is error!!!'.format(self.tokenization))
        
        return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., segment=1, tokenization='none', \
            afterNorm=False, decomp=None, d_out=None, input_len=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.decomp = decomp
        self.d_out = d_out

        if afterNorm:
            Norm = AfterNorm
        else:
            Norm = PreNorm

        att_dim = dim if input_len is None else input_len
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, Attention(att_dim, heads = heads, dim_head = att_dim, dropout = dropout, segment=segment, tokenization=tokenization)),
                Norm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

        if d_out is not None:
            self.projection = nn.Linear(dim, d_out)
            
    def forward(self, x):
        total_trend = torch.zeros_like(x) if self.decomp else None
        for attn, ff in self.layers:
            if self.decomp is not None:
                trend, x = self.decomp(x) #[B, C, L]
                total_trend += trend
            x = attn(x) + x
            x = ff(x) + x

        if self.decomp is not None:
            x = total_trend + x

        if self.d_out is not None:
            x = self.projection(x)
            
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., segment=1, query_segment=1, tokenization='time', \
            afterNorm=False, decomp=None, d_out=None, dec_self='att', dec_cross='att', input_len=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.decomp = decomp
        self.d_out = d_out

        if afterNorm:
            Norm = AfterNorm
        else:
            Norm = PreNorm

        
        att_dim = dim if input_len is None else input_len
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, Attention(att_dim, heads = heads, dim_head = att_dim, dropout = dropout, \
                    segment=query_segment, tokenization=tokenization)) \
                        if dec_self == 'att' else nn.Identity(),
                Norm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                Norm(dim, CrossAttention(att_dim, heads = heads, dim_head = att_dim, dropout = dropout, \
                    segment=query_segment, value_segment=segment, tokenization=tokenization)) \
                        if dec_cross == 'att' else nn.Identity(),
                Norm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

        if d_out is not None:
            self.projection = nn.Linear(dim, d_out)
            
    def forward(self, x, keys, values):
        total_trend = torch.zeros_like(x) if self.decomp else None
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

        if self.decomp is not None:
            x = total_trend + x

        if self.d_out is not None:
            x = self.projection(x)

        return x



