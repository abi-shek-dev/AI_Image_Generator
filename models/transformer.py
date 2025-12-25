import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    A simple "Transformer-like" layer (PixelCNN style) that predicts the next token
    based only on previous tokens.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        # Remove the extra padding at the end to keep size consistent (Causal)
        x = self.conv(x)
        return x[:, :, :-self.padding]

class PixelTransformer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, num_layers=8):
        super(PixelTransformer, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Stack of Causal Convolutions (Acting like a simplified Transformer)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    CausalConv1d(embedding_dim, embedding_dim, kernel_size=5, dilation=2**i),
                    nn.ReLU(True),
                    nn.BatchNorm1d(embedding_dim)
                )
            )
        
        self.output_head = nn.Conv1d(embedding_dim, num_embeddings, kernel_size=1)

    def forward(self, x):
        # x is a list of token indices: [Batch, Sequence_Length]
        x = self.embedding(x) # [Batch, Seq_Len, Dim]
        x = x.permute(0, 2, 1) # [Batch, Dim, Seq_Len]
        
        for layer in self.layers:
            x = x + layer(x) # Residual connection
            
        logits = self.output_head(x) # [Batch, Num_Embeddings, Seq_Len]
        return logits