import torch
from torch import nn
import math
from einops import rearrange
from .modules.time_layers import get_TimeLayer
from .modules.attention_layers import get_AttentionLayer
from .modules.Embed import get_embed
from .modules.decomp import series_decomp
# from .layers.decoder_layers import get_DecoderLayer


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

        # pos = pos.unsqueeze(1)
        # q = q + pos
        # k = k + pos

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, n_tokens, depth, heads, dim_head, timelayer, attentionlayer, \
        dropout = 0., Norm=PreNorm, decomp=None, *args, **kwargs):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([])
        self.decomp = decomp

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, get_AttentionLayer(attentionlayer, dim=dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                # Norm(dim, Attention(dim=dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                Norm(dim, get_TimeLayer(timelayer, dim=dim, n_tokens=n_tokens, dropout=dropout, *args, **kwargs)),
            ]))
            
    def forward(self, x): # [B, C, L]
        total_trend = torch.zeros_like(x) if self.decomp else None
        for attn, ff in self.layers:
            if self.decomp is not None:
                trend, x = self.decomp(x) #[B, C, L]
                total_trend += trend
            x = attn(x) + x #[B, C, L]
            x = ff(x) + x   #[B, C, L]
            
        if self.decomp is not None:
            x = total_trend + x
        return x


class TiTEvolution(nn.Module):
    def __init__(self, args):
        super(TiTEvolution, self).__init__()

        # series setting
        input_len, self.output_len, channels = args.seq_len, args.pred_len, args.enc_in
        # transformer structure setting
        num_layers, num_head, timelayer, attentionlayer = args.layer, args.n_heads, args.timelayer, args.attentionlayer
        d_model, mlp_dim, factor, activation = args.d_model, args.d_model, args.factor, args.activation
        # decomp and embed setting
        use_decomp, decomp_ks = args.use_decomp, args.decomp_ks
        pre_embed, self.s_scale = args.pre_embed, args.s_scale
        # weight init
        self.kaiming_normal, self.init_gain, pe_gain = args.kaiming_normal, args.init_gain, args.pe_gain
        # others
        dropout, output_attention, self.features = args.dropout, args.output_attention, args.features

        # time embedding
        # Embedding
        if args.dec_name == 'transformer':
            self.enc_embedding, self.dec_embeddingget_embed = get_embed(True, args.embedding, \
                c_in=args.enc_in, d_model=args.enc_in, embed_type=args.embed, freq=args.freq, dropout=args.dropout)
        elif args.dec_name == 'linear':
            self.enc_embedding, self.dec_embeddingget_embed = get_embed(False, args.embedding, \
                c_in=args.enc_in, d_model=args.enc_in, embed_type=args.embed, freq=args.freq, dropout=args.dropout)
        else:
            raise Exception("Type {} of decoder is error!!!".format(args.dec_name))

        # segmentation
        n_tokens = channels

        # vit embedding
        if args.pos_embedding:
            self.pos_embedding = nn.Parameter(pe_gain * torch.randn(1, n_tokens, d_model)) 
            # self.pos_embedding = nn.Parameter(pe_gain * torch.zeros(1, n_tokens, d_model)) 
            # self.pos_embedding = nn.Parameter(pe_gain * torch.randn(1, n_tokens, 1)) 
        else: 
            self.pos_embedding = None

        # pre_embedding
        self.embed_dropout = nn.Dropout(dropout)
        if pre_embed == 'linear':
            self.to_feature_embedding = nn.Linear(input_len, d_model, bias=True)
        elif pre_embed == 'shared_linear':
            device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"
            weights = nn.Parameter(0.01 * torch.randn((n_tokens, input_len, d_model), dtype=torch.float32, device=device))
            self.to_feature_embedding = lambda x : torch.einsum("bcl,cle->bce", x, weights)
        elif pre_embed == 'ffn':
            self.to_feature_embedding = get_TimeLayer(timelayer, dim=input_len, n_tokens=n_tokens, dropout=dropout, \
                mlp_dim=mlp_dim, factor=factor, output_attention=output_attention, activation=activation)
        elif pre_embed == 'none':
            self.to_feature_embedding = nn.Identity()
        else:
            raise Exception('Type {} of pre-embed is not implemented!!!'.format(pre_embed))

        # decomp
        if use_decomp:
            decomp = series_decomp(decomp_ks, True)
        else:
            decomp = None

        # model
        self.encoder = Transformer(dim=d_model, n_tokens=n_tokens, depth=num_layers, heads=num_head, dim_head=d_model, \
            timelayer=timelayer, attentionlayer=attentionlayer, \
            dropout=dropout, decomp=decomp, mlp_dim=mlp_dim, factor=factor, output_attention=output_attention, activation=activation)


        # self.decoder = nn.Linear(d_model, self.output_len, bias=False)
        if args.decoderlayer == 'linear':
            self.decoder = nn.Linear(d_model, self.output_len, bias=True)
        else:
            raise NotImplementedError(f'decoder layer type = {args.decoderlayer} is not implemented')

        # weight init
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.kaiming_normal:
                    std = math.sqrt(2. / (m.in_features + m.out_features)) * self.init_gain
                    m.weight.data.normal_(0., std)
                else:
                    std = 1. / m.in_features * self.init_gain
                    m.weight.data.normal_(0., std)
                # m.weight.data.normal_(0., 0.01)

    def forward(self, x, x_mark, dec_inp, y_mark): #[B, L, C]
        '''
        Input:
            x: shape == [B,L,C]
        '''
        x = x.permute(0,2,1) #[B, C, L]

        # time embedding
        x = self.enc_embedding(x.permute(0,2,1), x_mark).permute(0,2,1)
        # pre-embed
        x = self.to_feature_embedding(x)

        # # pos embedding
        if self.pos_embedding is not None:
            x += self.pos_embedding
        
        # embed dropout
        x = self.embed_dropout(x)

        # transformer
        if self.pos_embedding is not None:
            y = self.encoder(x, self.pos_embedding) #[B, C, L]
        else:
            y = self.encoder(x) #[B, C, L]

        # decoder
        # input of decoder is [B,C,L]
        y = self.decoder(y).permute(0,2,1) #[B, L, C]

        return y