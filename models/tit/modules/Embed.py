from pexpect import ExceptionPexpect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model//2]

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x): #[B, L, C]
        return self.embed(x) #[B, L, C]

class TimeFeatureEmbedding_Group(nn.Module):
    def __init__(self, channels, scale, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding_Group, self).__init__()

        self.channels = channels
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        # self.embed = nn.Linear(d_inp, d_model, bias=False)
        if scale:
            self.embed = nn.Conv1d(d_inp*channels, d_inp*channels, 1, stride=1, groups=channels, bias=False)
        else:
            self.embed = nn.Conv1d(d_inp*channels, channels, 1, stride=1, groups=channels, bias=False)

    def forward(self, x): #[B, C, L]
        x = x.repeat(1,self.channels,1)

        return self.embed(x) #[B, C, L]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_onlypos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_onlypos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=d_model) 
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        # self.temporal_embedding = TimeFeatureEmbedding_Group(channels=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # try:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # except:
        #     a = 1
        return self.dropout(x)

class DataEmbedding_onlyTimeFeature(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_onlyTimeFeature, self).__init__()

        # self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=d_model) 
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        # self.temporal_embedding = TimeFeatureEmbedding_Group(channels=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)



class DataEmbedding_wo_ve(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_ve, self).__init__()

        # self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model) 
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_onlyTimeStamp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_onlyTimeStamp, self).__init__()

        # self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=d_model) 
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.temporal_embedding(x_mark)
        return self.dropout(x)


def get_embed(need_dec, type_name, *args, **kwargs):
    if type_name == 'all':
        enc_embedding = DataEmbedding(*args, **kwargs)
        dec_embedding = DataEmbedding(*args, **kwargs) if need_dec else None
    elif type_name == 'te':
        enc_embedding = DataEmbedding_onlyTimeFeature(*args, **kwargs)
        dec_embedding = DataEmbedding_onlyTimeFeature(*args, **kwargs) if need_dec else None
    elif type_name == 'onlystamp':
        enc_embedding = DataEmbedding_onlyTimeStamp(*args, **kwargs)
        dec_embedding = DataEmbedding_onlyTimeStamp(*args, **kwargs) if need_dec else None
    elif type_name == 'wo_pos':
        enc_embedding = DataEmbedding_wo_pos(*args, **kwargs)
        dec_embedding = DataEmbedding_wo_pos(*args, **kwargs) if need_dec else None
    elif type_name == 'only_pos':
        enc_embedding = DataEmbedding_onlypos(*args, **kwargs)
        dec_embedding = DataEmbedding_onlypos(*args, **kwargs) if need_dec else None
    elif type_name == 'wo_ve':
        enc_embedding = DataEmbedding_wo_ve(*args, **kwargs)
        dec_embedding = DataEmbedding_wo_ve(*args, **kwargs) if need_dec else None
    elif type_name == 'none':
        enc_embedding = lambda x_enc, x_mark: x_enc
        dec_embedding = lambda x_enc, x_mark: x_enc if need_dec else None
    else:
        raise Exception("Type {} of type_name is error!!!".format(type_name))

    return enc_embedding, dec_embedding