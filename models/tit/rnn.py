from torch import nn

class RNNet(nn.Module):
    def __init__(self,num_layers,seq_len,pred_len, num_channels):
        super(RNNet, self).__init__()
        
        self.rnn=nn.RNN(
            input_size=num_channels,
            hidden_size=num_channels,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear=nn.Sequential(
            nn.Linear(in_features=seq_len,out_features=pred_len)
        )

    def forward(self, x):
        y,_ = self.rnn(x) 
        y = self.linear(y.permute(0,2,1))  
        y=y.permute(0,2,1)

        return y 