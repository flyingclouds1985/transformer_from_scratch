import torch 
from torch import nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        x size: (batch_size, seq_len)
        """
        emb = self.embedding(x)
        return emb
    
def test_token_embedding():
    vocab_size=30000
    emb_dim = 512
    batch_size = 2
    seq_len = 4
    batch_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    token_embedding = TokenEmbedding(vocab_size, emb_dim)
    output = token_embedding(batch_token_ids)
    
    print("output shape: ", output.shape)
    expected_shape = (batch_size, seq_len, emb_dim)
    assert expected_shape == output.shape,  \
    f"output sentor shape mismatch, Expected {expected_shape}, but got {output.shape}"


if __name__ == "__main__":
    test_token_embedding()