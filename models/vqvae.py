import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    The 'Codebook' layer. It maps continuous features to the nearest
    'word' in the codebook dictionary.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings # Size of dictionary (e.g., 512 words)
        self.embedding_dim = embedding_dim   # Size of each word vector
        self.commitment_cost = commitment_cost

        # The Dictionary itself (learnable)
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # Flatten inputs
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances between inputs and codebook vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
        # Find nearest codebook index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize (Snap to nearest)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)
        
        # Loss: Pull codebook to inputs and inputs to codebook
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-Through Estimator (Copy gradients across the discrete layer)
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized, encoding_indices

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        )
    def forward(self, x):
        return x + self._block(x)

class VQVAE(nn.Module):
    def __init__(self, num_hiddens=128, num_residual_hiddens=32, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        # 1. Encoder (Compresses Image)
        self._encoder = nn.Sequential(
            nn.Conv2d(3, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1),
            ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens),
            ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)
        )
        # Pre-Quantization convolution
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        
        # 2. Vector Quantizer (The "Token" Maker)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # 3. Decoder (Rebuilds Image from Tokens)
        self._decoder = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1),
            ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens),
            ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hiddens // 2, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity