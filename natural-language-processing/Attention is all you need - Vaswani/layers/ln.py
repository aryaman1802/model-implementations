import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """This is the 'norm' part in the 'add & norm' block in the paper."""
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        # instead of simply doing self.alpha = torch.ones(1)
        # we use nn.Parameter() so that when we call the state dict 
        # of the model, then we can see this alpha
        # if we only use torch.ones(), then we won't see alpha in 
        # the model's state dict
        self.alpha = nn.Parameter(torch.ones(features))  # multiplied
        self.beta = nn.Parameter(torch.zeros(features))  # added
        
    def forward(self, x):
        # apply mean after the batch dimension
        # mean usually cancels the dimension to which it is applied,  
        # but we want to keep it
        mean = x.mean(dim=-1, keepdim=True)
        # similarly for standard deviation
        std = x.std(dim=-1, keepdim=True)
        # apply the layer normalization
        fraction = (x - mean) / (torch.sqrt(std**2 + self.eps))
        x_normalized = self.alpha * fraction + self.beta
        return x_normalized
    
    