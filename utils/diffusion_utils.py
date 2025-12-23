import torch

class DiffusionEngine:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Define the noise schedule (how much noise to add at each step)
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        """
        Forward Process: Adds Gaussian noise to image x at time step t.
        Returns: (Noisy Image, The Noise Added)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        
        epsilon = torch.randn_like(x) # Random Gaussian Noise
        
        # The formula: Mean + Variance
        noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        
        return noisy_image, epsilon

    def sample_timesteps(self, n):
        """Generates random time steps for training"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)