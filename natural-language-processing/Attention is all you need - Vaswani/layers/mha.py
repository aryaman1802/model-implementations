import torch
import torch.nn as nn

# Second way of doing multihead attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        # Recall that d_model = num_heads * d_k   OR   d_k = d_model // num_heads
        assert d_model == (self.num_heads * self.d_k), "d_model must be equal \
            to self.num_heads * self.d_k"
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        x = x.transpose(1, 2)
        # shape of x becomes: [batch_size, self.num_heads, seq_length, self.d_k]
        return x
        
        
    def combine_heads(self, x):
        """
        x: tensor of shape [batch_size, num_heads, seq_length, d_k]

        Returns a tensor of shape [batch_size, seq_length, d_model]

        Description:
        Combine the multiple heads back to original shape, ie,
        we want the shape of x to be [batch_size, seq_length, d_model]
        """
        # After calling split_heads function, the shape of x became:
        # [batch_size, self.num_heads, seq_length, self.d_k]
        batch_size, num_heads, seq_length, d_k = x.size()
        
        assert num_heads == self.num_heads, "Number of heads must be equal to self.num_heads"
        assert d_k == self.d_k, "d_k must be equal to self.d_k"
        assert self.d_model == (num_heads * d_k), "d_model must be equal to \
            self.num_heads * self.d_k"
        
        x = x.transpose(1, 2).contiguous()
        # shape of x becomes: [batch_size, seq_length, self.num_heads, self.d_k]
        x = x.view(batch_size, seq_length, self.d_model)
        return x
        

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations to Q, K, V
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads 
        heads_combined = self.combine_heads(attn_output)
        
        # Apply output transformation
        output = self.W_o(heads_combined)
        return output
    
