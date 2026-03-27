import torch
from torch import nn
import numpy as np

class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention block from the Attention Is All You Need.
    """
    def __init__(self, d_model:int= 512, h:int= 8):
        super().__init__()
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.h = h
        self.d_k = d_model//h

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch, seq_len, self.h, self.d_k)
        K = K.view(batch, seq_len, self.h, self.d_k)
        V = V.view(batch, seq_len, self.h, self.d_k)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        scores = torch.softmax(torch.matmul(Q,K.transpose(-2, -1))/torch.sqrt(torch.tensor(self.d_k, dtype = torch.float32)), dim = -1)
        attention = torch.matmul(scores , V)
        attention = attention.transpose(1,2).contiguous()
        attention = attention.view(batch, seq_len, d_model)

        multi_head = self.W_O(attention)

        return multi_head


class FeedForward(nn.Module):
    def __init__(self, d_model:int= 512, d_ff:int = 2048):
        super().__init__()
        self.feed = nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.ReLU(),
                        nn.Linear(d_ff, d_model)
                        )

    def forward(self, x):
        x = self.feed(x)
    
        return x
        

class EncoderBlock(nn.Module):
    def __init__(self, d_model:int= 512, h:int = 8):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.MultiHead = MultiHeadAttention(d_model, h)
        self.FF = FeedForward(d_model)

    def forward(self, x):
        x = x + self.MultiHead(x)
        x = self.norm1(x)

        x = x + self.FF(x)
        x = self.norm2(x)

        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
