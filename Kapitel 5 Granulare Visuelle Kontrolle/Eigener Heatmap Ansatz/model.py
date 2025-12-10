
class IdentityEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super(IdentityEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.latent_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.fc(x)
        return x



class IdentityDecoder(nn.Module):
    def __init__(self, latent_dim=1036, out_channels=3):
        super(IdentityDecoder, self).__init__()
        self.latent_dim = latent_dim

        # Fully connected layer to reshape latent into 4x4x512
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512 * 4 * 4),
            nn.ReLU()
        )

        # Decoder: mirror of encoder with ConvTranspose2d
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()

        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)   # Reshape to 4x4 feature map
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x



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