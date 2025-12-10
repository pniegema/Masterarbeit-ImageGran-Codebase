import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapEncoder64(nn.Module):
    def __init__(self, in_channels=68, latent_dim=128, first_layer_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, first_layer_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(first_layer_channels),
            nn.ReLU(True),

            # 32x32 -> 16x16
            nn.Conv2d(first_layer_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),

            # 4x4 -> 2x2
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)

        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, latent_dim)
        self.scaling = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)         # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)           # (B, latent_dim)
        x = self.scaling(x)
        return x


class MappingNetwork(nn.Module):
    """Maps latent codes to intermediate style space"""

    def __init__(self, latent_dim, style_dim, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = latent_dim if i == 0 else style_dim
            layers.extend([
                nn.Linear(in_dim, style_dim),
                nn.LeakyReLU(0.2)
            ])
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


class AttentionGuidedAdaIN(nn.Module):
    """Layer-aware attention-guided AdaIN for blending identity and motion styles"""

    def __init__(self, style_dim, num_features, layer_idx=0, num_layers=5):
        super().__init__()
        self.num_features = num_features
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        # Identity AdaIN params
        self.fc_gamma_id = nn.Linear(style_dim, num_features)
        self.fc_beta_id = nn.Linear(style_dim, num_features)

        # Motion AdaIN params
        self.fc_gamma_motion = nn.Linear(style_dim, num_features)
        self.fc_beta_motion = nn.Linear(style_dim, num_features)

        # Layer encoding: normalized layer depth [0, 1]
        layer_encoding = torch.tensor([layer_idx / max(num_layers - 1, 1)], dtype=torch.float32)
        self.register_buffer('layer_encoding', layer_encoding)

        # Attention gate with layer awareness
        # Input: [w_identity, w_motion, layer_encoding]
        self.attention_gate = nn.Sequential(
            nn.Linear(style_dim * 2 + 1, num_features * 2),
            nn.ReLU(),
            nn.Linear(num_features * 2, num_features),
            nn.Sigmoid()
        )

    def forward(self, x, w_identity, w_motion):
        """
        x: Feature map (B, C, H, W)
        w_identity: Identity style embedding (B, style_dim)
        w_motion: Motion style embedding (B, style_dim)

        Returns attention-weighted blended features
        """
        B, C, H, W = x.size()

        # Calculate AdaIN parameters from both styles
        gamma_id = self.fc_gamma_id(w_identity)  # (B, C)
        beta_id = self.fc_beta_id(w_identity)  # (B, C)

        gamma_motion = self.fc_gamma_motion(w_motion)  # (B, C)
        beta_motion = self.fc_beta_motion(w_motion)  # (B, C)

        # Prepare attention input with layer context
        layer_enc = self.layer_encoding.expand(B, 1)
        att_input = torch.cat([w_identity, w_motion, layer_enc], dim=1)

        # Compute attention weights (per-channel)
        # att_weights close to 1 → prefer identity
        # att_weights close to 0 → prefer motion
        att_weights = self.attention_gate(att_input)  # (B, C)

        # Blend AdaIN parameters using attention
        gamma = att_weights * gamma_id + (1 - att_weights) * gamma_motion
        beta = att_weights * beta_id + (1 - att_weights) * beta_motion

        # Instance Normalization
        mean = x.view(B, C, -1).mean(dim=2, keepdim=True).view(B, C, 1, 1)
        std = x.view(B, C, -1).std(dim=2, keepdim=True).view(B, C, 1, 1) + 1e-5
        x_norm = (x - mean) / std

        # Reshape for broadcasting
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        # Apply modulation
        out = gamma * x_norm + beta

        return out


class ResidualGenBlock(nn.Module):
    """Generator block with dual-style attention-guided AdaIN"""

    def __init__(self, style_dim, in_channels, out_channels, layer_idx=0, num_layers=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain1 = AttentionGuidedAdaIN(style_dim, out_channels, layer_idx, num_layers)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.adain2 = AttentionGuidedAdaIN(style_dim, out_channels, layer_idx, num_layers)
        self.lrelu2 = nn.LeakyReLU(0.2)

        # Residual connection
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = None

    def forward(self, x, w_identity, w_motion):
        residual = x
        if self.res_conv:
            residual = self.res_conv(residual)

        x = self.conv1(x)
        x = self.adain1(x, w_identity, w_motion)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.adain2(x, w_identity, w_motion)
        x = self.lrelu2(x)

        out = x + residual
        return out


class StyleGAN1Generator(nn.Module):
    """StyleGAN generator with separate identity and motion control"""

    def __init__(self, identity_dim=512, motion_dim=384, style_dim=512):
        super().__init__()

        # Separate mapping networks for identity and motion
        self.identity_mapping = MappingNetwork(identity_dim, style_dim)
        self.motion_mapping = MappingNetwork(motion_dim, style_dim)

        # Learnable constant input (4x4)
        self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Progressive upsampling with layer-aware attention
        # Layer 0 (early): Should prefer identity (global appearance)
        # Layer 4 (late): Should prefer motion (facial details)
        num_layers = 5
        self.progression = nn.ModuleList([
            ResidualGenBlock(style_dim, 512, 512, layer_idx=0, num_layers=num_layers),  # 4x4
            ResidualGenBlock(style_dim, 512, 512, layer_idx=1, num_layers=num_layers),  # 8x8
            ResidualGenBlock(style_dim, 512, 256, layer_idx=2, num_layers=num_layers),  # 16x16
            ResidualGenBlock(style_dim, 256, 128, layer_idx=3, num_layers=num_layers),  # 32x32
            ResidualGenBlock(style_dim, 128, 64, layer_idx=4, num_layers=num_layers),   # 64x64
            #ResidualGenBlock(style_dim, 64, 32, layer_idx=5, num_layers=num_layers),    # 128x128
        ])

        self.to_rgb = nn.Conv2d(64, 3, 1)

    def forward(self, identity_embed, motion_vector):
        """
        identity_embed: (B, identity_dim) - from buffalo_sc or similar
        motion_vector: (B, motion_dim) - concatenated pose/expression/mouth vectors

        Returns: (B, 3, 64, 64) RGB image in range [-1, 1]
        """
        # Map to style space
        w_identity = self.identity_mapping(identity_embed)
        w_motion = self.motion_mapping(motion_vector)

        # Start with constant input
        x = self.constant.repeat(identity_embed.size(0), 1, 1, 1)  # (B, 512, 4, 4)

        # First block at 4x4 (no upsampling)
        x = self.progression[0](x, w_identity, w_motion)

        # Progressive upsampling + generation
        for block in self.progression[1:]:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = block(x, w_identity, w_motion)

        # Convert to RGB
        out = self.to_rgb(x)
        out = torch.tanh(out)  # [-1, 1] range

        return out

    def get_attention_weights(self, identity_embed, motion_vector):
        """
        Utility function to visualize attention weights per layer during training.
        Returns list of attention weights for each layer.
        """
        w_identity = self.identity_mapping(identity_embed)
        w_motion = self.motion_mapping(motion_vector)

        attention_weights = []
        for block in self.progression:
            # Get attention from first AdaIN in block
            layer_enc = block.adain1.layer_encoding.expand(identity_embed.size(0), 1)
            att_input = torch.cat([w_identity, w_motion, layer_enc], dim=1)
            att = block.adain1.attention_gate(att_input)
            attention_weights.append(att.mean().item())  # Average across batch and channels

        return attention_weights

class StyleGAN1Generator_256(nn.Module):
    """StyleGAN generator with separate identity and motion control"""

    def __init__(self, identity_dim=512, motion_dim=384, style_dim=512):
        super().__init__()

        # Separate mapping networks for identity and motion
        self.identity_mapping = MappingNetwork(identity_dim, style_dim)
        self.motion_mapping = MappingNetwork(motion_dim, style_dim)

        # Learnable constant input (4x4)
        self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Progressive upsampling with layer-aware attention
        # Layer 0 (early): Should prefer identity (global appearance)
        # Layer 4 (late): Should prefer motion (facial details)
        num_layers = 7
        self.progression = nn.ModuleList([
            ResidualGenBlock(style_dim, 512, 512, layer_idx=0, num_layers=num_layers),  # 4x4
            ResidualGenBlock(style_dim, 512, 512, layer_idx=1, num_layers=num_layers),  # 8x8
            ResidualGenBlock(style_dim, 512, 384, layer_idx=2, num_layers=num_layers),  # 16x16
            ResidualGenBlock(style_dim, 384, 256, layer_idx=3, num_layers=num_layers),  # 32x32
            ResidualGenBlock(style_dim, 256, 128, layer_idx=4, num_layers=num_layers),   # 64x64
            ResidualGenBlock(style_dim, 128, 64, layer_idx=5, num_layers=num_layers),    # 128x128
            ResidualGenBlock(style_dim, 64, 32, layer_idx=6, num_layers=num_layers),  # 256x256
        ])

        self.to_rgb = nn.Conv2d(32, 3, 1)

    def forward(self, identity_embed, motion_vector):
        """
        identity_embed: (B, identity_dim) - from buffalo_sc or similar
        motion_vector: (B, motion_dim) - concatenated pose/expression/mouth vectors

        Returns: (B, 3, 64, 64) RGB image in range [-1, 1]
        """
        # Map to style space
        w_identity = self.identity_mapping(identity_embed)
        w_motion = self.motion_mapping(motion_vector)

        # Start with constant input
        x = self.constant.repeat(identity_embed.size(0), 1, 1, 1)  # (B, 512, 4, 4)

        # First block at 4x4 (no upsampling)
        x = self.progression[0](x, w_identity, w_motion)

        # Progressive upsampling + generation
        for block in self.progression[1:]:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = block(x, w_identity, w_motion)

        # Convert to RGB
        out = self.to_rgb(x)
        out = torch.tanh(out)  # [-1, 1] range

        return out

    def get_attention_weights(self, identity_embed, motion_vector):
        """
        Utility function to visualize attention weights per layer during training.
        Returns list of attention weights for each layer.
        """
        w_identity = self.identity_mapping(identity_embed)
        w_motion = self.motion_mapping(motion_vector)

        attention_weights = []
        for block in self.progression:
            # Get attention from first AdaIN in block
            layer_enc = block.adain1.layer_encoding.expand(identity_embed.size(0), 1)
            att_input = torch.cat([w_identity, w_motion, layer_enc], dim=1)
            att = block.adain1.attention_gate(att_input)
            attention_weights.append(att.mean().item())  # Average across batch and channels

        return attention_weights

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = StyleGAN1Generator_256(
        identity_dim=512,  # buffalo_sc embedding size
        motion_dim=384,  # pose_exp(256) + mouth(128)
        style_dim=512
    )

    # Dummy inputs
    batch_size = 4
    identity_embed = torch.randn(batch_size, 512)  # From buffalo_sc
    motion_vector = torch.randn(batch_size, 384)  # From your heatmap encoders

    # Generate
    output = generator(identity_embed, motion_vector)
    print(f"Output shape: {output.shape}")  # (4, 3, 64, 64)

    # Visualize attention weights
    att_weights = generator.get_attention_weights(identity_embed, motion_vector)
    print("\nAttention weights per layer (higher = more identity, lower = more motion):")
    for i, w in enumerate(att_weights):
        print(f"Layer {i} (resolution {4 * 2 ** i}x{4 * 2 ** i}): {w:.3f}")

    params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {params/1e6:.2f}M")
