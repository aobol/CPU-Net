'''
This script contains the PositionalUNet network along with 3 candidate discriminators:
* RNN+Attention discriminator
* CNN+PositionalEncoding Discriminator
* Fully Connected Discriminators
we have tested all 3 discriminators, turns out that the RNN+Attention works the best
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import math
from dataset import SEQ_LEN


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=11, padding=5,bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=7, padding=3,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
            # torch.nn.LeakyReLU(),
            # torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, start=0, dropout=0.1, max_len=10000,factor=1.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.factor = factor

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)
        self.start = start
    # @torchsnooper.snoop()
    def forward(self, x):
        x = x + self.factor*self.pe[:,:,self.start:(self.start+x.size(2))]
        x = self.dropout(x)
        return x
    
class PositionalUNet(nn.Module):
    def __init__(self):
        super(PositionalUNet, self).__init__()
        self.bilinear = True
        
        multi = 40
        
        self.inc = DoubleConv(1, multi)
        self.down1 = Down(multi, multi*2)
        self.down2 = Down(multi*2, multi*4)
        self.down3 = Down(multi*4, multi*8)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(multi*8, multi*16 // factor)
        
        self.fc_mean = torch.nn.Conv1d(multi*16 // factor, multi*16 // factor,1)
        self.fc_var = torch.nn.Conv1d(multi*16 // factor, multi*16 // factor,1)
        
        self.up1 = Up(multi*16, multi*8 // factor, self.bilinear)
        self.up2 = Up(multi*8, multi*4 // factor, self.bilinear)
        self.up3 = Up(multi*4, multi*2 // factor, self.bilinear)
        self.up4 = Up(multi*2, multi // factor, self.bilinear)
        self.outc = OutConv(multi // factor, 1)
        
        self.pe1 = PositionalEncoding(multi)
        self.pe2 = PositionalEncoding(multi*2)
        self.pe3 = PositionalEncoding(multi*4)
        self.pe4 = PositionalEncoding(multi*8)
        self.pe5 = PositionalEncoding(multi*16//factor)
        self.pe6 = PositionalEncoding(multi*8// factor,start=multi*4)
        self.pe7 = PositionalEncoding(multi*4// factor,start=multi*2)
        self.pe8 = PositionalEncoding(multi*2// factor,start=multi*2)
        self.pe9 = PositionalEncoding(multi// factor,start=0,factor=1.0)
        
    
    def reparametrize(self, mu,logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(mu)
        return eps.mul(std).add_(mu)
    
    # @torchsnooper.snoop()
    def forward(self, x):        
        x1 = self.pe1(self.inc(x))
        x2 = self.pe2(self.down1(x1))
        x3 = self.pe3(self.down2(x2))
        x4 = self.pe4(self.down3(x3))
        x5 = self.down4(x4)
        x5 = self.pe5(self.reparametrize(self.fc_mean(x5), self.fc_var(x5)))
        
        
        x = self.pe6(self.up1(x5, x4))
        x = self.pe7(self.up2(x, x3))
        x = self.pe8(self.up3(x, x2))
        x = self.up4(x, x1)
        output = self.outc(x)
        out = []
        
        # Normalize the output waveforms to interval between [0,1]
        # for ibatch in range(output.size(0)):
        #     out.append(((output[ibatch,0] - output[ibatch,0].min()) / (output[ibatch,0].max() - output[ibatch,0].min())).unsqueeze(0).unsqueeze(0))
        # output = torch.cat(out,dim=0)
        
        return output
    
    
#The RNN based model:
class RNN(nn.Module):
    def __init__(self,get_attention = False):
        super(RNN, self).__init__()
        
        bidirec = True    #Whether to use a bidirectional RNN
        self.bidirec =bidirec
        feed_in_dim = 128
        self.seg = 1      #Segment waveform to reduce its length. If the original waveform is (2000,1), then segment it with self.seg=5 can reduce its length to (400,5)
        self.emb_dim = 64
        self.emb_tick = 1/1000.0
        self.embedding = nn.Embedding(int(1/self.emb_tick),self.emb_dim)
        self.seq_len = (SEQ_LEN)//self.seg
        if bidirec:
            self.RNNLayer = torch.nn.GRU(input_size = self.emb_dim, hidden_size = feed_in_dim//2,num_layers=1, batch_first=True,bidirectional=True,dropout=0.0)
            feed_in_dim *= 2
        else:
            self.RNNLayer = torch.nn.GRU(input_size = self.emb_dim, hidden_size = feed_in_dim//2,num_layers=1, batch_first=True,bidirectional=False,dropout=0.0)
        self.attention_weight = nn.Linear(feed_in_dim//2, feed_in_dim//2, bias=False)
        self.norm = torch.nn.BatchNorm1d(feed_in_dim//2)
        self.get_attention = get_attention
        
        fc1, fc2 = (feed_in_dim, int(feed_in_dim*0.25))
        do = 0.2
        self.fcnet = nn.Linear(fc1, 1)

    def forward(self, x):
        x = x.view(-1,self.seq_len)
        x = (x - x.min(dim=-1,keepdim=True)[0])/(x.max(dim=-1,keepdim=True)[0] - x.min(dim=-1,keepdim=True)[0])
        x = (x/self.emb_tick).long()
        x = self.embedding(x)
        bsize = x.size(0)
        output, hidden = self.RNNLayer(x)
        if self.bidirec:
            hidden =  hidden[-2:]
            hidden = hidden.transpose(0,1).reshape(bsize,-1)
        else:
            hidden =  hidden[-1]
        
        attention_scores = self.calculate_attention_scores(output, hidden)
        
        if self.get_attention:
            return attention_scores  # Return attention scores if get_attention flag is True

        # Apply attention scores
        context = torch.sum(attention_scores.unsqueeze(-1).expand_as(output) * output, dim=1)
        x = self.fcnet(torch.cat([context, hidden], dim=-1))
        return torch.sigmoid(x)
    
    def calculate_attention_scores(self, output, hidden):
        """Calculate attention scores."""
        inner_product = torch.einsum("ijl,il->ij", output, hidden)
        output_norm = torch.linalg.norm(output, dim=-1)
        hidden_norm = torch.linalg.norm(hidden, dim=-1, keepdim=True)
        attention_scores = torch.softmax(inner_product / (output_norm * hidden_norm + 1e-8), dim=-1)
        return attention_scores

    def get_attention_weights(self, x):
        """A method to get attention weights explicitly."""
        self.get_attention_flag = True  # Ensure the model returns attention scores
        attention_weights = self.forward(x)
        self.get_attention_flag = False  # Reset the flag
        return attention_weights