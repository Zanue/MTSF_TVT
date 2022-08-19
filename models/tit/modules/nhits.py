import torch
from torch.nn import functional as F
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat
from torchsummary import summary


import einops
from .layers.time_layers import get_TimeLayer
from .layers.attention_layers import get_AttentionLayer
from .layers.embed import get_TimeEmbedding
from .layers.decomp import series_decomp
# from .layers.decoder_layers import get_DecoderLayer


class MLP(nn.Module):
    def __init__(self,len1,len2,num_channels,num_layers,dropout=0.1,activation='relu'):
        super().__init__()
        mlp = []

        for _ in range(num_layers):
            mlp.append(nn.Linear(len1, len1))

            if activation == 'relu':
                mlp.append(nn.ReLU())
            else:
                raise NotImplementedError(f'activation = {activation} not implemented')

            # mlp.append(Rearrange('B C L -> B L C'))
            # mlp.append(nn.LayerNorm(num_channels))
            # mlp.append(Rearrange('B L C -> B C L'))

            # mlp.append(nn.LayerNorm(len1))


            mlp.append(nn.Dropout(dropout))

        mlp.append(nn.Linear(len1, len2))

        self.mlp = nn.Sequential(*mlp)
    
    def forward(self,x):
        '''
        x.shape = [B,C,L1]
        return.shape = [B,C,L2]
        '''
        return self.mlp(x)


class MultiRateSampling(nn.Module):
    def __init__(self,kernel_size,mode='max'):
        assert kernel_size % 2 == 1, f'kernel_size = {kernel_size} should be odd'
        super().__init__()
        padding = (kernel_size - 1)//2
        if mode == 'max':
            self.pool = nn.MaxPool1d(kernel_size=kernel_size,stride=1,padding=padding)
        elif mode == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=kernel_size,stride=1,padding=padding)
        else:
            raise NotImplementedError(f'mode = {mode} not implemented')
    
    def forward(self,x):
        '''
        x.shape = [B,C,L]
        return.shape = [B,C,L]
        '''
        return self.pool(x)


class NHitsBlock(nn.Module):
    def __init__(
        self,
        observation_len,
        prediction_len,
        reduced_observation_len,
        reduced_prediction_len,
        hidden_len,
        kernel_size,
        num_channels,
        num_layers,
        multi_rate_sampling_mode='max',
        dropout=0.1,
        activation='relu',
        interpolate_mode='linear'
    ) -> None:
        super().__init__()

        self.prediction_len   = prediction_len
        self.observation_len  = observation_len
        self.interpolate_mode = interpolate_mode

        self.multi_rate_sampling = MultiRateSampling(kernel_size, multi_rate_sampling_mode)
        self.to_hidden = MLP(observation_len, hidden_len, num_channels, num_layers, dropout, activation)
        self.to_reduced_forecast =  nn.Linear(hidden_len, reduced_prediction_len)
        # self.to_reduced_backcast =  nn.Linear(hidden_len, reduced_observation_len)
        self.to_reduced_backcast =  nn.Linear(hidden_len, observation_len)
    
    def forward(self,x):
        '''
        Input:
            x.shape = [B,C,observation_len]
        Output:
            forecast, shape = [B,C,prediction_len]
            x - backcast, shape = [B,C,observation_len]
        '''
        x_pooling = self.multi_rate_sampling(x)
        hidden = self.to_hidden(x_pooling)
        reduced_forecast = self.to_reduced_forecast(hidden)
        reduced_backcast = self.to_reduced_backcast(hidden)

        forecast = F.interpolate(reduced_forecast, size=self.prediction_len,  mode=self.interpolate_mode, align_corners=False)
        # backcast = F.interpolate(reduced_backcast, size=self.observation_len, mode=self.interpolate_mode, align_corners=False)
        backcast = reduced_backcast
        return forecast, x - backcast


class NHitsModel(nn.Module):
    def __init__(
        self,
        observation_len:int,
        prediction_len:int,
        reduced_observation_len_each_block:list,
        reduced_prediction_len_each_block :list,
        hidden_len_each_block :list,
        kernel_size_each_block:list,
        num_channels:int,
        num_layers_each_block:list,
        multi_rate_sampling_mode:str = 'max',
        dropout:float                = 0.1,
        activation:str               = 'relu',
        interpolate_mode:str         = 'linear',
        *args, **kwds
    ) -> None:

        super().__init__()
        assert len(reduced_observation_len_each_block) == len(reduced_prediction_len_each_block) == \
            len(hidden_len_each_block) == len(kernel_size_each_block) == len(num_layers_each_block), \
                f'Length of each list should equal'

        self.prediction_len = prediction_len
        self.num_blocks     = len(reduced_observation_len_each_block)
        self.model = nn.ModuleList()
        for i in range(self.num_blocks):
            self.model.append(
                NHitsBlock(
                    observation_len,
                    prediction_len,
                    reduced_observation_len_each_block[i],
                    reduced_prediction_len_each_block[i],
                    hidden_len_each_block[i],
                    kernel_size_each_block[i],
                    num_channels,
                    num_layers_each_block[i],
                    multi_rate_sampling_mode,
                    dropout,
                    activation,
                    interpolate_mode
                )
            )
        

        for m in self.model.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.normal_(0.01)
        
    def forward(self,x):
        '''
        Input:
            x.shape = [B,C,observation_len]
        Output:
            forecast, shape = [B,C,prediction_len]
        '''
        batch_size   = x.shape[0]
        num_channels = x.shape[1]
        forecast = torch.zeros(batch_size,num_channels,self.prediction_len).to(x.device)

        for i in range(self.num_blocks):
            f,x = self.model[i](x)
            forecast = forecast + f
        
        return forecast



class NHits(nn.Module):
    def __init__(self, args):
        '''NHits forecast as encoder'''
        super().__init__()

        # series setting
        input_len, self.output_len, channels = args.seq_len, args.pred_len, args.enc_in
        # transformer structure setting
        num_layers, num_head, timelayer, attentionlayer = args.layer, args.n_heads, args.timelayer, args.attentionlayer
        d_model, mlp_dim, factor = input_len, input_len, args.factor
        # decomp and embed setting
        use_decomp, decomp_ks = args.use_decomp, args.decomp_ks
        use_embedding, self.embed_method, fixed_embed = args.use_embedding, args.embed_method, args.fixed_embed
        pre_embed, self.s_scale = args.pre_embed, args.s_scale
        # others
        dropout, output_attention, self.features = args.dropout, args.output_attention, args.features

        # time embedding
        if use_embedding:
            self.upscale, self.downscale, self.time_embed, n_tokens = get_TimeEmbedding(channels, self.embed_method, fixed_embed)
        else:
            self.time_embed, n_tokens = None, channels
        
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
                mlp_dim=mlp_dim, factor=factor, output_attention=output_attention)
        elif pre_embed == 'none':
            self.to_feature_embedding = nn.Identity()
        else:
            raise Exception('Type {} of pre-embed is not implemented!!!'.format(pre_embed))



        # encoder
        # zerz
        # kernel_size
        num_blocks = 3
        self.encoder = NHitsModel(
            observation_len = input_len,
            prediction_len = self.output_len,
            reduced_observation_len_each_block = torch.linspace(input_len//2, input_len, num_blocks).int().tolist(),
            reduced_prediction_len_each_block = [self.output_len//4, self.output_len//2, self.output_len//1],
            hidden_len_each_block = [96] * num_blocks,
            kernel_size_each_block = [7,3,1],                
            num_channels = -1,
            num_layers_each_block = [2] * num_blocks
        )


        # self.decoder = nn.Linear(d_model, self.output_len, bias=False)
        # zerz
        if args.decoderlayer == 'linear':
            self.decoder = nn.Linear(d_model, self.output_len, bias=True)
        elif args.decoderlayer == 'nhits':
            num_blocks = 3
            self.decoder = NHitsModel(
                observation_len = self.output_len,
                prediction_len = self.output_len,
                reduced_observation_len_each_block = torch.linspace(input_len//2, input_len, num_blocks).int().tolist(),
                reduced_prediction_len_each_block = torch.linspace(self.output_len//2, self.output_len, num_blocks).int().tolist(),
                hidden_len_each_block = [512] * num_blocks,
                kernel_size_each_block = [7,3,1],                
                num_channels = -1,
                num_layers_each_block = [2] * num_blocks
            )
        elif args.decoderlayer == 'empty':
            self.decoder = nn.Identity()
        else:
            raise NotImplementedError(f'decoder layer type = {args.decoderlayer} is not implemented')

        # weight init
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.005)

    def forward(self, x, x_mark, dec_inp, y_mark): #[B, L, C]
        '''x.shape == [B,L,C]'''

        x = x.permute(0,2,1) #[B, C, L]

        # time embedding -> upscale
        if self.time_embed is not None:
            x = self.upscale(x) + self.time_embed(x_mark.permute(0,2,1)) #[B, C, L]

        # pre-embed
        x = self.to_feature_embedding(x)
        x = self.embed_dropout(x)

        # transformer
        y = self.encoder(x) #[B, C, L]

        # time embedding -> downscale
        if self.time_embed is not None:
            y = self.downscale(y)

        # decoder
        # input of decoder is [B,C,L]
        y = self.decoder(y).permute(0,2,1) #[B, L, C]

        return y
        
if __name__ == "__main__":
    pass
    # batch_size = 7
    # channel = 4
    # len1 = 16
    # len2 = 32

    # num_blocks = 3

    # x = torch.randn(batch_size,channel,len1).cuda()

    # m = NHitsModel(
    #     observation_len                    = len1,
    #     prediction_len                     = len2,
    #     reduced_observation_len_each_block = [len1 // 2]*num_blocks,
    #     reduced_prediction_len_each_block  = [len2 // 2]*num_blocks,
    #     hidden_len_each_block              = [(len1+len2) // 2]*num_blocks,
    #     kernel_size_each_block             = [5]*num_blocks,
    #     num_channels                       = channel,
    #     num_layers_each_block              = [4]*num_blocks,
    # ).cuda()

    # f = m(x)
    # print(f.shape)

    # summary(m,input_size=(channel,len1))

    # for i in m.children():
    #     print(i)


        