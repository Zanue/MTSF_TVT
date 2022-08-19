from torch import nn
import torch
import numpy as np
from .modules.Embed import DataEmbedding, DataEmbedding_wo_pos


class Linear(nn.Module):
    def __init__(self, input_len, output_len, num_layers=1, middle=None):
        super(Linear, self).__init__()
        linear = [nn.Linear(input_len, output_len)]
        
        for _ in range(num_layers-1):
            linear.append(nn.ReLU())
            linear.append(nn.Linear(input_len, output_len))

        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        return self.linear(x.permute(0,2,1)).permute(0,2,1)



class MLP_ChannelTokens(nn.Module):
    def __init__(self, input_len, output_len, middle=None):
        super(MLP_ChannelTokens, self).__init__()

        if middle is None:
            middle = output_len
        self.linear1 = nn.Sequential(
            nn.Linear(input_len, middle),
            nn.GELU(),
            nn.Linear(middle, middle)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(middle, middle),
            nn.GELU(),
            nn.Linear(middle, output_len)
        )

        self.skip = nn.Linear(input_len, middle)
        self.act = nn.GELU()
        # self.fc = nn.Linear(output_len, output_len)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.act(self.linear1(x) + self.skip(x))
        # y = self.act(self.linear1(x))
        y = self.linear2(y)
        # y = self.fc(y)

        return y.permute(0,2,1)

class MLP_TimeTokens(nn.Module):
    def __init__(self, input_len, output_len, num_channels, middle=None):
        super(MLP_TimeTokens, self).__init__()

        if middle is None:
            middle = num_channels
        self.linear1 = nn.Sequential(
            nn.Linear(num_channels, middle),
            nn.GELU(),
            nn.Linear(middle, middle)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(middle, middle),
            nn.GELU(),
            nn.Linear(middle, num_channels)
        )

        self.skip = nn.Linear(num_channels, middle)
        self.act = nn.GELU()

        self.fc = nn.Linear(num_channels, num_channels)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.act(self.linear1(x) + self.skip(x))
        # y = self.act(self.linear1(x))
        y = self.linear2(y)
        y = self.fc(y)

        return y


class MlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle=None):
        super(MlpBlock, self).__init__()
        if middle is None:
            middle =  (input_dim + output_dim) // 2
        self.fc1 = nn.Linear(input_dim, middle)
        self.fc2 = nn.Linear(middle, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, channels_dim, tokens_dim, middle=None):
        super(MixerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels_dim)
        self.mlp_token_mixing = MlpBlock(tokens_dim, tokens_dim)
        self.norm2 = nn.LayerNorm(channels_dim)
        self.mlp_channel_mixing = MlpBlock(channels_dim, channels_dim)

    def forward(self, x):
        y = self.norm1(x) #[B, C, L]
        y = y.permute(0,2,1) #[B, L, C]
        y = self.mlp_token_mixing(y) #[B, L, C]
        y = y.permute(0,2,1) #[B, C, L]
        x = x + y
        y = self.norm2(x) #[B, C, L]
        y = x + self.mlp_channel_mixing(y) #[B, C, L]

        return y


class MLPMixer(nn.Module):
    def __init__(self, input_len, output_len, num_channels, num_blocks, middle=None, d_model=512, \
        embed='timeF', freq='h', dropout=0.):
        super(MLPMixer, self).__init__()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(MixerBlock(input_len, num_channels))
        self.blocks = nn.Sequential(*blocks)
        # self.norm = nn.LayerNorm(64)
        self.fc = nn.Linear(input_len, output_len)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):

        x = x.permute(0,2,1) #[B, C, L]
        y = self.blocks(x) #[B, C, L]
        y = self.fc(y) #[B, C, L]

        return y.permute(0,2,1)


class MLPMixer_embedding(nn.Module):
    def __init__(self, input_len,label_len, output_len, num_channels, num_blocks, middle=None, use_embedding=False, d_model=512, \
        embed='timeF', freq='h', dropout=0.):
        super(MLPMixer_embedding, self).__init__()

        self.pred_len = output_len
        self.use_embedding = use_embedding
        if use_embedding:
            input_len = label_len + output_len
            self.embedding = DataEmbedding_wo_pos(num_channels, d_model, embed, freq, dropout)
        else:
            self.upscale = nn.Linear(num_channels, d_model)
        self.downscale = nn.Linear(d_model, num_channels)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(MixerBlock(input_len, d_model))
        self.blocks = nn.Sequential(*blocks)
        # self.norm = nn.LayerNorm(64)
        self.fc = nn.Linear(input_len, output_len)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        if self.use_embedding:
            x_dec[:,-self.pred_len:, :] = x_dec[:,-self.pred_len:, :] + torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            x = self.embedding(x_dec, x_mark_dec)
        else:
            x = self.upscale(x)

        x = x.permute(0,2,1) #[B, C, L]
        y = self.blocks(x) #[B, C, L]

        # y = self.downscale(y.permute(0,2,1)).permute(0,2,1)
        if self.use_embedding:
            y = self.fc(y) #[B, C, L]
        else:
            y = self.fc(y) #[B, C, L]
            y = y[:, :, -self.pred_len:]

        return y.permute(0,2,1)