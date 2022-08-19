import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.Embed import get_embed
from .modules.TransformerLayer import Transformer, TransformerDecoder
from utils.rank_comp import cal_ratios



class Transformer_TimeTokens_noSeg(nn.Module):
    def __init__(self, args):
        super(Transformer_TimeTokens_noSeg, self).__init__()
        
        self.pred_len = args.pred_len
        self.dec_name = args.dec_name
        self.output_attention = args.output_attention
        self.init_gain = args.init_gain

        # Embedding
        if args.dec_name == 'transformer':
            self.enc_embedding, self.dec_embedding = get_embed(True, args.embedding, \
                c_in=args.enc_in, d_model=args.d_model, embed_type=args.embed, freq=args.freq, dropout=args.dropout)
        elif args.dec_name == 'linear':
            self.enc_embedding, self.dec_embedding = get_embed(False, args.embedding, \
                c_in=args.enc_in, d_model=args.d_model, embed_type=args.embed, freq=args.freq, dropout=args.dropout)
        else:
            raise Exception("Type {} of decoder is error!!!".format(args.dec_name))

        d_model = args.d_model
        d_out = args.c_out if args.c_out != d_model else None
        # structure, encoder
        self.encoder = Transformer(dim=d_model, depth=args.e_layers, heads=args.n_heads, dim_head=d_model, \
            mlp_dim=d_model, dropout=args.dropout, output_attention=self.output_attention)
        # structure, decoder
        if args.dec_name == 'transformer':
            self.decoder = TransformerDecoder(dim=d_model, depth=args.d_layers, heads=args.n_heads, dim_head=d_model, \
                mlp_dim=d_model, dropout=args.dropout, d_out=d_out, dec_self=args.dec_self, dec_cross=args.dec_cross, output_attention=self.output_attention)
        elif args.dec_name == 'linear':
            self.decoder = nn.Linear(args.seq_len, args.pred_len)
        else:
            raise Exception("Type {} of decoder is error!!!".format(args.dec_name))

        # weight init
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # std = 1. / m.in_features * self.init_gain
                # m.weight.data.normal_(0., std)
                m.weight.data.normal_(0, 0.01)

    def forward(self, x, x_mark, dec_inp, y_mark, ratio=False): #[B, L, C]
        
        enc_in = self.enc_embedding(x, x_mark)
        if not self.output_attention:
            enc_out = self.encoder(enc_in)
        else:
            enc_out, enc_atts = self.encoder(enc_in)

        if self.dec_name == 'transformer':
            dec_in = self.dec_embedding(dec_inp, y_mark)
            if not self.output_attention:
                dec_out = self.decoder(dec_in, enc_out, enc_out)
            else:
                dec_out, dec_self_atts, dec_cross_atts = self.decoder(dec_in, enc_out, enc_out)
        elif self.dec_name == 'linear':
            dec_in = enc_out.permute(0,2,1)
            dec_out = self.decoder(dec_in).permute(0,2,1)

        if self.output_attention:
            if not ratio:
                return dec_out[:, -self.pred_len:, :], (enc_atts, dec_self_atts, dec_cross_atts)
            else:
                ratio_enc_in = cal_ratios(enc_in) #[B, L, C]
                ratio_enc_out = cal_ratios(enc_out) #[B, L, C]
                ratio_dec_in = cal_ratios(dec_in) #[B, L, C]
                ratio_dec_out = cal_ratios(dec_out) #[B, L, C]

                return dec_out[:, -self.pred_len:, :], (enc_atts, dec_self_atts, dec_cross_atts), \
                    (ratio_enc_in, ratio_enc_out, ratio_dec_in, ratio_dec_out)

        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]



class Transformer_ChannelTokens_noSeg(nn.Module):
    def __init__(self, args):
        super(Transformer_ChannelTokens_noSeg, self).__init__()

        self.pred_len = args.pred_len
        self.dec_name = args.dec_name
        self.output_attention = args.output_attention
        self.learnable_query = args.learnable_query
        self.init_gain = args.init_gain
        self.batch_size = args.batch_size
        self.channels = args.enc_in

        # Embedding
        if args.dec_name == 'transformer':
            self.enc_embedding, self.dec_embedding = get_embed(True, args.embedding, \
                c_in=args.enc_in, d_model=args.enc_in, embed_type=args.embed, freq=args.freq, dropout=args.dropout)
        elif args.dec_name == 'linear':
            self.enc_embedding, self.dec_embedding = get_embed(False, args.embedding, \
                c_in=args.enc_in, d_model=args.enc_in, embed_type=args.embed, freq=args.freq, dropout=args.dropout)
        else:
            raise Exception("Type {} of decoder is error!!!".format(args.dec_name))

        # structure, encoder
        encoder_out = args.label_len + args.pred_len if args.dec_name == 'transformer' else None
        self.encoder = Transformer(dim=args.seq_len, depth=args.e_layers, heads=args.n_heads, dim_head=args.seq_len, \
            mlp_dim=args.seq_len, dropout=args.dropout, d_out=encoder_out, output_attention=self.output_attention)
        # structure, decoder
        if args.dec_name == 'transformer':
            # self.enc_projection = nn.Linear(args.seq_len, args.label_len + args.pred_len)
            self.decoder = TransformerDecoder(dim=encoder_out, depth=args.d_layers, heads=args.n_heads, dim_head=encoder_out, \
                mlp_dim=encoder_out, dropout=args.dropout, dec_self=args.dec_self, dec_cross=args.dec_cross, output_attention=self.output_attention)
        elif args.dec_name == 'linear':
            self.decoder = nn.Linear(args.seq_len, args.pred_len)
        else:
            raise Exception("Type {} of decoder is error!!!".format(args.dec_name))

        if self.learnable_query:
            device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"
            self.queries = nn.Parameter(0.01 * torch.randn((args.batch_size, args.seq_len, args.enc_in), \
                dtype=torch.float32, requires_grad=True, device=device))

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                std = 1. / m.in_features * self.init_gain
                m.weight.data.normal_(0., std)
                # m.weight.data.normal_(0., 0.01)


    def forward(self, x, x_mark, dec_inp, y_mark, ratio=False): #[B, L, C]
        
        enc_in = self.enc_embedding(x, x_mark).permute(0,2,1)

        if not self.output_attention:
            enc_out = self.encoder(enc_in)
        else:
            enc_out, enc_atts = self.encoder(enc_in)


        if self.dec_name == 'transformer':
            dec_in = self.dec_embedding(dec_inp, y_mark)
            if self.learnable_query:
                dec_in = dec_in + self.queries
            dec_in = dec_in.permute(0,2,1)
            if not self.output_attention:
                dec_out = self.decoder(dec_in, enc_out, enc_out)
            else:
                dec_out, dec_self_atts, dec_cross_atts = self.decoder(dec_in, enc_out, enc_out)
            dec_out = dec_out.permute(0,2,1)
        elif self.dec_name == 'linear':
            dec_in = enc_out
            dec_out = self.decoder(dec_in).permute(0,2,1)

        if self.output_attention:
            if not ratio:
                if self.dec_name == 'transformer':
                    return dec_out[:, -self.pred_len:, :], (enc_atts, dec_self_atts, dec_cross_atts)
                else:
                    return dec_out[:, -self.pred_len:, :], None
            else:
                ratio_enc_in = cal_ratios(enc_in) #[B, C, L]
                ratio_enc_out = cal_ratios(enc_out) #[B, C, L]
                ratio_dec_in = cal_ratios(dec_in) #[B, C, L]
                ratio_dec_out = cal_ratios(dec_out.permute(0,2,1)) #[B, C, L]

                if self.dec_name == 'transformer':
                    return dec_out[:, -self.pred_len:, :], (enc_atts, dec_self_atts, dec_cross_atts), \
                        (ratio_enc_in, ratio_enc_out, ratio_dec_in, ratio_dec_out)
                else:
                    return dec_out[:, -self.pred_len:, :], None, \
                        (ratio_enc_in, ratio_enc_out, ratio_dec_in, ratio_dec_out)

        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]



