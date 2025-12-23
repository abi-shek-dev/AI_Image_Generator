import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    """A helper block that keeps image information stable (Residual Connection)"""
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels=3, num_residuals=9):
        super(Generator, self).__init__()
        
        # 1. Initial Block (Process the input image)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # 2. Downsampling (Shrink image to find features)
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # 3. Transformer (ResNet Blocks) - The "Magic" happens here
        # This part swaps the 'Horse' features for 'Zebra' features
        for _ in range(num_residuals):
            model += [ResNetBlock(in_features)]

        # 4. Upsampling (Enlarge image back to original size)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # 5. Output Layer (Convert back to RGB colors)
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, img_channels, kernel_size=7),
            nn.Tanh() # Forces values between -1 and 1 (Standard for images)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)