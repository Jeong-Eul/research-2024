import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='replicate')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x # B * N,  patch num, d_model

class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input):
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1) # B, N, T + Padding
        return output 

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model).float()

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching shape of x is B, N, T
        n_vars = x.shape[1] # N
        x = self.padding_patch_layer(x) # B, N, T + Padding
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # B, N, patch num, patch len
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # B * N, patch num, patch len
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars
    
class PromptPatchEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, target_len):
        super(PromptPatchEmbedding, self).__init__()
        
        kernel_size = 100
        stride = 1
        padding = 0 
        self.dim = target_len
        intermediate_dim = 1024 

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=intermediate_dim, 
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=intermediate_dim, out_channels=intermediate_dim, 
                               kernel_size=1) 
        self.conv3 = nn.Conv1d(in_channels=intermediate_dim, out_channels=intermediate_dim, 
                               kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=intermediate_dim, out_channels=output_dim, 
                               kernel_size=1)

    def forward(self, x):
        B_N, L, D = x.shape
        x = x.transpose(1, 2)
        x = self.conv1(x)  # (B * N, D, L) -> (B * N, intermediate_dim, P)
        x = self.conv2(x)  # (B * N, intermediate_dim, P) -> (B * N, intermediate_dim, P)
        x = self.conv3(x)  # (B * N, intermediate_dim, P) -> (B * N, intermediate_dim, P)
        x = self.conv4(x)  # (B * N, intermediate_dim, P) -> (B * N, output_dim, P)

        x = x.transpose(1, 2)[:, :self.dim, :]
  
        return x