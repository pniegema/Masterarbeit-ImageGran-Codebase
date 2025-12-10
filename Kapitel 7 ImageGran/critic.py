import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """ConvNeXt-like residual block used in R3GAN critic."""
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.dwconv = nn.Conv2d(out_channels, out_channels, 7, 1, 3, groups=out_channels)
        self.norm = nn.GroupNorm(1, out_channels)
        self.linear = nn.Conv2d(out_channels, out_channels, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.down = nn.AvgPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        residual = self.down(self.skip(x))
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.linear(x)
        x = self.down(x)
        return (x + residual) * (1 / (2 ** 0.5))

class R3GANCritic(nn.Module):
    """
    R3GAN-inspired critic with ConvNeXt-like blocks for 256x256 images.
    Outputs scalar realness scores (no sigmoid).
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(in_channels, base_channels, downsample=True),           # 256 → 128
            ConvBlock(base_channels, base_channels * 2, downsample=True),     # 128 → 64
            ConvBlock(base_channels * 2, base_channels * 4, downsample=True), # 64 → 32
            ConvBlock(base_channels * 4, base_channels * 8, downsample=True), # 32 → 16
            ConvBlock(base_channels * 8, base_channels * 8, downsample=True), # 16 → 8
            ConvBlock(base_channels * 8, base_channels * 8, downsample=True), # 8 → 4
        ])

        self.final = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.final(x)
        return x.view(x.size(0))

def compute_gradients(y, x):
    """Utility: gradient of scalar outputs y wrt inputs x."""
    grad = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grad

def gradient_penalty_R1(D, real_images, gamma=10.0):
    """R1 penalty on real samples."""
    real_images.requires_grad_(True)
    real_scores = D(real_images)
    grads = compute_gradients(real_scores.sum(), real_images)
    grad_penalty = grads.view(grads.size(0), -1).pow(2).sum(1).mean()
    return 0.5 * gamma * grad_penalty

def gradient_penalty_R2(D, fake_images, gamma=10.0):
    """R2 penalty on fake samples."""
    fake_images.requires_grad_(True)
    fake_scores = D(fake_images)
    grads = compute_gradients(fake_scores.sum(), fake_images)
    grad_penalty = grads.view(grads.size(0), -1).pow(2).sum(1).mean()
    return 0.5 * gamma * grad_penalty


# ====== RpGAN (Relativistic-paired) Loss ======

def rpgan_discriminator_loss(D, real_images, fake_images):
    """R³GAN critic loss (RpGAN formulation)."""
    d_real = D(real_images)
    d_fake = D(fake_images.detach())
    # Relativistic paired comparison
    loss_real = F.softplus(-(d_real - d_fake)).mean()
    loss_fake = F.softplus(d_fake - d_real).mean()
    return loss_real + loss_fake

def rpgan_generator_loss(D, real_images, fake_images):
    """R³GAN generator loss (symmetric relativistic form)."""
    d_real = D(real_images.detach())
    d_fake = D(fake_images)
    loss_real = F.softplus(d_real - d_fake).mean()
    loss_fake = F.softplus(-(d_fake - d_real)).mean()
    return loss_real + loss_fake


# ====== Full R³GAN Losses ======

def r3gan_d_loss(D, real_images, fake_images, gamma=15.0):
    """Total R3GAN critic loss: RpGAN + R1 + R2 penalties."""
    base_loss = rpgan_discriminator_loss(D, real_images, fake_images)
    r1 = gradient_penalty_R1(D, real_images, gamma)
    r2 = gradient_penalty_R2(D, fake_images, gamma)
    return base_loss + r1 + r2, {'base': base_loss.item(), 'r1': r1.item(), 'r2': r2.item()}


def r3gan_d_loss_fast(D, real_images, fake_images, lambda_r1=10.0, lambda_r2=10.0, do_reg=False):
    """
    R3GAN loss wie in Figure 2.
    do_reg: Wenn False, skip R1+R2 (lazy regularization)
    """
    # Base relativistic loss
    base_loss = rpgan_discriminator_loss(D, real_images, fake_images)

    # Lazy regularization
    r1_loss = 0.0
    r2_loss = 0.0

    if do_reg:
        r1_loss = gradient_penalty_R1(D, real_images, gamma=lambda_r1)
        r2_loss = gradient_penalty_R2(D, fake_images, gamma=lambda_r2)

    total_loss = base_loss + r1_loss + r2_loss

    return total_loss, {
        'base': base_loss.item(),
        'r1': r1_loss.item() if isinstance(r1_loss, torch.Tensor) else 0.0,
        'r2': r2_loss.item() if isinstance(r2_loss, torch.Tensor) else 0.0
    }

def r3gan_g_loss(D, real_images, fake_images):
    """Total R3GAN generator loss."""
    return rpgan_generator_loss(D, real_images, fake_images)


def r3gan_d_loss_alternating(D, real_images, fake_images, lambda_r1=10.0, lambda_r2=10.0, do_r1=False, do_r2=False):
    """
    R3GAN mit alternierenden R1/R2 penalties (effizienter).

    Args:
        do_r1: Wenn True, berechne R1 penalty
        do_r2: Wenn True, berechne R2 penalty
    """
    # Base relativistic loss
    base_loss = rpgan_discriminator_loss(D, real_images, fake_images)

    r1_loss = 0.0
    r2_loss = 0.0

    # R1 penalty (nur wenn do_r1=True)
    if do_r1:
        r1_loss = gradient_penalty_R1(D, real_images, gamma=lambda_r1)

    # R2 penalty (nur wenn do_r2=True)
    if do_r2:
        r2_loss = gradient_penalty_R2(D, fake_images, gamma=lambda_r2)

    total_loss = base_loss + r1_loss + r2_loss

    return total_loss, {
        'base': base_loss.item(),
        'r1': r1_loss.item() if isinstance(r1_loss, torch.Tensor) else 0.0,
        'r2': r2_loss.item() if isinstance(r2_loss, torch.Tensor) else 0.0
    }

if __name__ == "__main__":
    # Simple test for 256x256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = R3GANCritic().to(device)

    real_imgs = torch.randn(4, 3, 256, 256).to(device)
    fake_imgs = torch.randn(4, 3, 256, 256).to(device)

    d_loss, loss_dict = r3gan_d_loss(D, real_imgs, fake_imgs)
    g_loss = r3gan_g_loss(D, real_imgs, fake_imgs)

    print("Discriminator Loss:", d_loss.item(), loss_dict)
    print("Generator Loss:", g_loss.item())
    print("Critic Parameters:", sum(p.numel() for p in D.parameters()), "parameters")