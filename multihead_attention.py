import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_w = nn.Linear(d_model, d_model)
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)
        self.o_w = nn.Linear(d_model, d_model)


    def forward(self, query, key, value, casual_mask=None, pad_mask=None):
        """
        query, key, value are of size (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)


        # (batch_size,seq_len, num_hueads, head_dim) ->  (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        # (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn, v)
        merged = attn_output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output = self.o_w(merged)
        return output
        

def test_multihead_attention():
    # Define the dimensions
    embed_dim = 8  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    seq_length = 5  # Length of the sequence

    # Create an instance of MultiheadAttention
    multihead_attn = MultiHeadAttention(embed_dim, num_heads)

    # Simulate a batch of token embeddings (batch size = 1)
    token_embeddings = torch.randn(1, seq_length, embed_dim)

    # Apply the MultiheadAttention layer (self-attention)
    output = multihead_attn(token_embeddings, token_embeddings, token_embeddings)

    print("Input Embeddings:")
    print(token_embeddings)
    print("\nOutput of Multihead Attention:")
    print(output)
    

if __name__ == "__main__":
    test_multihead_attention()