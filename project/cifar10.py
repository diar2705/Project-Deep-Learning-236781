import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Main branch
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut branch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.res(x) + self.shortcut(x)
        return F.gelu(out)

class Encoder(nn.Module):
    def __init__(self, latent_dim=128, input_channels=3):
        super(Encoder, self).__init__()
        c_hid = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            nn.GELU(),
            ResidualBlock(c_hid, c_hid),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            nn.GELU(),
            ResidualBlock(2*c_hid, 2*c_hid),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            nn.GELU(),
            ResidualBlock(2*c_hid, 2*c_hid),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super(Decoder, self).__init__()
        c_hid = 32  # Base number of channels
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),  # Expand to 4x4 feature map
            nn.Unflatten(1, (2 * c_hid, 4, 4)),    # Reshape to (2*c_hid, 4, 4)
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2, output_padding=1),  # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),  # No spatial change
            nn.GELU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, padding=1, stride=2, output_padding=1),  # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),  # No spatial change
            nn.GELU(),
            nn.ConvTranspose2d(c_hid, output_channels, kernel_size=3, padding=1, stride=2, output_padding=1),  # 16x16 => 32x32
            nn.Tanh()  # Output in range [-1, 1] for image data
        )
    
    def forward(self, x):
        return self.decoder(x)

class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),  # Increase width
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)  # Output layer
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(x)

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
    
class Enclassifier(nn.Module):
    def __init__(self, latent_dim=128):
        super(Enclassifier, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.classifier = Classifier()

    def forward(self, x):
        z = self.encoder(x)
        logits, probas = self.classifier(z)
        return logits, probas