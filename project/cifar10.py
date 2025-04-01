import torch
import torch.nn as nn
import torch.nn.functional as F

def init_conv(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.main.apply(init_conv)
        self.shortcut.apply(init_conv)

    def forward(self, x):
        out = self.main(x)
        return F.silu(out + self.shortcut(x))


class Encoder(nn.Module):
    def __init__(self, latent_dim=128, input_channels=3):
        super(Encoder, self).__init__()
        c_hid = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(c_hid),
            nn.SiLU(),
            ResidualBlock(c_hid, c_hid),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(2 * c_hid),
            nn.SiLU(),
            ResidualBlock(2 * c_hid, 2 * c_hid),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(4 * c_hid),
            nn.SiLU(),
            ResidualBlock(4 * c_hid, 4 * c_hid),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(4 * 4 * 4 * c_hid, latent_dim),
        )
        self.encoder.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super(Decoder, self).__init__()
        c_hid = 64  
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 4 * c_hid),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Unflatten(1, (4 * c_hid, 4, 4)),
            ResidualBlock(4 * c_hid, 4 * c_hid),
            nn.ConvTranspose2d(4 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(2 * c_hid),
            nn.SiLU(),
            ResidualBlock(2 * c_hid, 2 * c_hid),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(c_hid),
            nn.SiLU(),
            ResidualBlock(c_hid, c_hid),
            nn.ConvTranspose2d(c_hid, output_channels, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.Tanh(),
        )
        self.decoder.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)
        self.decoder.apply(init_conv)

    def forward(self, x):
        return self.decoder(x)


class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.SiLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(x)
