import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.skip(residual)  # Skip connection
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, latent_dim=128, input_channels=3):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2),  # Output: (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output: (64, 15, 15)
            ResidualBlock(64, 128, stride=2),  # Output: (128, 8, 8)
            ResidualBlock(128, 256, stride=2),  # Output: (256, 4, 4)
            nn.Flatten(),  # Output: (256 * 4 * 4 = 4096)
            nn.Linear(256 * 4 * 4, latent_dim),  # Output: latent_dim
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super(Decoder, self).__init__()
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),  # Output: (256 * 4 * 4)
            nn.LeakyReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Output: (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # Output: (256, 8, 8)
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # Output: (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # Output: (128, 16, 16)
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # Output: (64, 32, 32)
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),  # Output: (3, 32, 32)
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.decoder_fc(x)
        x = x.view(-1, 256, 4, 4)  # Reshape to (batch_size, 256, 4, 4)
        x = self.decoder_conv(x)
        return x

class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Approximate GELU as ReLU
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        

    def forward(self, x):
        logits = self.classifier(x)
        probas = nn.functional.softmax(logits, dim=1)
        return logits, probas

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    