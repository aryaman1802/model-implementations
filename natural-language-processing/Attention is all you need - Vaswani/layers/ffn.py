import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    """Implementation of the FFN equation."""
    def __init__(self, d_model = 512, d_ff = 2048, dropout: float = 0.0):
        super(FeedForwardNetwork, self).__init__()
        # the shapes are mentioned in the paper
        # I have written them above for reference
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)
        self.relu = nn.ReLU()
        # if dropout is zero, then nn.Dropout does not do anything
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.relu(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        return x
     