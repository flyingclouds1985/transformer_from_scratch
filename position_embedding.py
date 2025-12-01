import math
import torch 
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        max_len: max sequenence length
        d_model: model dimention / token embedding dimention
        """
        super(PositionEmbedding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)

        # pe(pos, 2i) = sin( pos / 10000^{i2/d_model}  )
        # pe(pos, 2i+1) = cos( pos / 10000^{i2/d_model}  )
        # exp(log (10000^{i2 / d_model})   ) = exp ( (2i / d_model) *log (10000) )
        pos = torch.arange(0, max_len, 1).float().unsqueeze(1)
        # div_term of size (d_model/2, )
        div_term = torch.exp(torch.arange(0, d_model, step=2).float() * (- math.log(10000) ) / d_model )
        print("div_term shape: ", div_term.shape)
        print("pos * div_term shape: ", (pos * div_term).shape)

        pe[:, 0::2] = torch.exp(pos * div_term)
        pe[:, 1::2] = torch.exp(pos * div_term)
        self.register_buffer('pe', pe)


        
    
    def forward(self, x):
        """
        x size : (batch_size, seq_len, d_model)
        output: (1, seq_len, d_model)
        """
        seq_len = x.size(1)
        # print("self.pe[:seq_len, :] shape: ", self.pe[:seq_len, :].shape)
        # y = x + self.pe[:seq_len, :]
        # print("y shape: ", y.shape)
        # print("equal or not:", torch.equal(y, x + self.pe[:seq_len, :].unsqueeze(0)))

        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x

def test_position_embedding():
    max_len=10
    d_model = 512
    batch_size = 2
    seq_len = 4

    x = torch.randn(batch_size, seq_len, d_model)
    position_embedding = PositionEmbedding(d_model, max_len)

    output = position_embedding(x)

    print("Input shape: ", x.shape)
    print("output shape: ", output.shape)

if __name__ == "__main__":
    test_position_embedding()