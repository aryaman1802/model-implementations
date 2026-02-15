import torch
import torch.nn as nn
from mha import MultiHeadAttention
from ffn import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # multi-head attention
        # target mask to ignore certain parts of the decoder's input
        selfattn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(selfattn_output))
        # masked multi-head attention
        # query from decoder
        # key and value from encoder
        # source mask is used to ignore certain parts of the encoder's output
        crossattn_output = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(crossattn_output))
        # feed forward network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x
    