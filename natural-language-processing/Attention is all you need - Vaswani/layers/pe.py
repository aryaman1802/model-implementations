import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Applies the sin-cos positional encoding to the input and output embedding.
    max_seq_len: The maximum length of the sequence for which positional encodings 
    are pre-computed.
    d_model: dimension of the input
    
    Returns: A tensor of shape (batch_size, max_seq_len, d_model)
    """
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        # initialize the positional encoding matrix of size (max_seq_len, d_model) 
        # with zeros
        pe = torch.zeros(max_seq_len, d_model)
        # this is the 'pos' in the formula
        # we want it to be of shape (max_seq_len, 1)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        # this is the 'i' in the formula
        # no need to multiply by 2, because we are considering only even numbers
        i = torch.arange(0, d_model, 2)
        # below we implement the numerically more stable version of the sin and cos
        # positional encoding (you can check the PE_derivation.png for the derivation)
        denominator = torch.exp(-(i/d_model) * math.log(10000.0))
        # apply sin positional encoding to the even indices
        pe[:, 0::2] = torch.sin(pos * denominator)
        # apply cos positional encoding to the odd indices
        pe[:, 1::2] = torch.cos(pos * denominator)
        # we need to add the batch dimension so that we can apply it to 
        # batches of sentences
        pe = pe.unsqueeze(0)  # new shape: (1, max_seq_len, d_model)
        # pe is registered as a buffer, which means it will be part of the module's 
        # state but will not be considered a trainable parameter.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # we don't want to train the positional encoding, ie, we don't want to make it
        # a learnable parameter, so we set its requires_grad to False
        # shape of x is (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x
    