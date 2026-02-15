import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.dec_lay import DecoderLayer
from layers.enc_lay import EncoderLayer
from layers.ffn import FeedForwardNetwork
from layers.mha import MultiHeadAttention
from layers.pe import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                 num_layers, d_ff, max_seq_len, dropout=0.0):
        """
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: The dimensionality of the model's embeddings.
        num_heads: Number of attention heads in the multi-head attention mechanism.
        num_layers: Number of layers for both the encoder and the decoder.
        d_ff: Dimensionality of the inner layer in the feed-forward network.
        max_seq_len: Maximum sequence length for positional encoding.
        dropout: Dropout rate for regularization.
        """
        super(Transformer, self).__init__()
        self.enc_emb = nn.Embedding(src_vocab_size, d_model)
        self.dec_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(max_seq_len, d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
            ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
            ])
        
        self.proj = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def generate_mask(self, src, tgt):
        """
        This method is used to create masks for the source and target sequences, 
        ensuring that padding tokens are ignored and that future tokens are not 
        visible during training for the target sequence.
        """
        # Create a mask for the source sequence (src)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # Create a mask for the target sequence (tgt)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        # Determine the length of the target sequence
        seq_length = tgt.size(1)
        # Create a no-peak mask for the target sequence to prevent peeking into future tokens
        # create a vector of ones of shape (1, seq_length, seq_length)
        ones_vector = torch.ones(1, seq_length, seq_length)
        # Create an upper triangular mask starting above the diagonal (i.e., diagonal=1). 
        # This results in a matrix where elements above the diagonal are set to one, and 
        # elements on and below the diagonal are zero.
        peak_mask = torch.triu(ones_vector, diagonal=1)
        # Invert the peak mask to create a no-peak mask
        nopeak_mask = (1 - peak_mask)
        # Convert the nopeak mask to a boolean mask
        nopeak_mask = nopeak_mask.bool()
        # Combine the padding mask and the no-peak mask for the target sequence
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask


    def forward(self, src, tgt):
        """
        This method defines the forward pass for the Transformer, taking source 
        and target sequences and producing the output predictions. Below we outline
        the steps involved in the forward pass:
        1. Input Embedding and Positional Encoding: The source and target sequences 
        are first embedded using their respective embedding layers and then added to 
        their positional encodings.
        2. Encoder Layers: The source sequence is passed through the encoder layers, 
        with the final encoder output representing the processed source sequence.
        3. Decoder Layers: The target sequence and the encoder's output are passed 
        through the decoder layers, resulting in the decoder's output.
        4. Final Linear Layer: The decoder's output is mapped to the target vocabulary 
        size using a fully connected (linear) layer.
        
        Output: The final output is a tensor representing the model's predictions for 
        the target sequence.
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.pos_enc(self.enc_emb(src)))
        tgt_embedded = self.dropout(self.pos_enc(self.dec_emb(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.proj(dec_output)
        # calculate output probabilities
        output_prob = F.softmax(output, dim=-1)
        return output_prob
    
    
    
    

