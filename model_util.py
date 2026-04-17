import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        ## Build positional encoder layer
        # Define dropout layer with the given dropout probability (Hint: Use torch.nn.Dropout)
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)

        # divisor term for positional encoding (i.e., 1 / (10000 ^ (2i/d))) (Hint: Use torch.arange(0, d_model, 2))
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)

        # Multiply position index with divisor term, and then apply sine to even elements and cosine to odd elements.
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.pe = pe
        self.pe.requires_grad = False


    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Add positional encodings to the input token embeddings
        x = x + self.pe[:x.size(0), :, :]

        # Apply dropout
        x = self.dropout(x)

        return x


def positionalEncodingTest():
    # test your function
    print("\nImplementing Positional Encoding using PyTorch ")
    pe = PositionalEncoding(100, 0.0)
    if (pe(torch.zeros((1, 1, 100))).squeeze()[:10] - torch.tensor([0., 1., 0., 1., 0., 1., 0., 1., 0., 1.])).abs().sum().item() < 0.0001:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)

    if (pe(torch.zeros((10, 1, 100))).squeeze()[-1][:10] - torch.tensor([0.4121, -0.9111, 0.9330, 0.3599, -0.0567, 0.9984, -0.8931, 0.4498, -0.9192, -0.3938])).abs().sum().item() < 0.001:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)
