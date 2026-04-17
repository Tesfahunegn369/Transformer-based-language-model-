import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model_util import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout, device):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.device = device

        ### START CODE HERE (Replace instances of 'None' with your own code) ###
        # Define the positional encoder with embedding dimension and dropout probability
        self.pos_encoder =  PositionalEncoding(d_model, dropout)

        # Define transformer encoder layer with embedding dimension, num_heads, dimension of hidden layers, and dropout (Hint: Use torch.nn.TransformerEncoderLayer)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        # Define transformer encoder model with transformer encoder layer with number of layers (Hint: User torch.nn.TransformerEncoder)
        # Note: Residual connection and Layer normalization are already implemented in TransformerEncoder. You don't have to implement them.
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Define Embedding layer (Hint: Use torch.nn.Embedding)
        self.embedding = nn.Embedding(ntoken, d_model)

        # Define Linear layer (Feed-forward layer) (Hint: Use torch.nn.Linear)
        self.linear = nn.Linear(d_model, ntoken)

        ### END CODE HERE ###

        # Initialize the model parameters
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    # Forward propagation
    def forward(self, src, src_mask=None):
        """
        Arguments:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
                if none, square subsequent mask for auto-regressive training

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        # Input token embeddings
        src = self.embedding(src) * math.sqrt(self.d_model)

        ### START CODE HERE (Replace instances of 'None' with your own code) ###
        # Apply positional encoder to add positional encoding
        src = self.pos_encoder(src)

        if src_mask is None: ## Do not change "None" in this line. It should remain as is.
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)

        # Apply transformer encoder with src_mask
        output = self.transformer_encoder(src, src_mask)

        # Apply point-wise feed-forward network to transform the tensor with shape (..., d_model) to (..., ntoken)
        output = self.linear(output)

        # ### END CODE HERE ###

        return output
