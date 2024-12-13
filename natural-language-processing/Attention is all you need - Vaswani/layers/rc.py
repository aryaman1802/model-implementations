import torch
import torch.nn as nn
from ln import LayerNormalization

class ResidualConnection(nn.Module):
    """This is the 'add' part in the 'add & norm' block."""
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super(ResidualConnection, self).__init__()
        # if dropout is zero, then nn.Dropout does not do anything
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, sublayer):
        """
        x: input
        sublayer: different layers of the transformer architecture (eg: multi-head
        attention, feed-forward network, etc.), we will pass these layers as
        functions to this class.
        
        Returns the skip or residual connection.
        """
        # most implementations first do normalization and then pass x to the sublayer
        # we will also do this way
        return x + self.dropout(sublayer(self.norm(x)))
        # however, the paper first passes x to the sublayer and then does the norm
        # return x + self.dropout(self.norm(sublayer(x)))
    
    