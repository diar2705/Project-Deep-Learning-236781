import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),  # Output: (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output: (64, 15, 15)
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # Output: (192, 15, 15)
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output: (192, 7, 7)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Output: (384, 7, 7)
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Output: (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Output: (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),  # Output: (256 * 7 * 7 = 12544)
            nn.Linear(256 * 7 * 7, 4096),  # Output: 4096
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),  # Output: 1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, latent_dim),  # Output: latent_dim
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),  # Output: (512)
            nn.ReLU(),
            nn.Linear(512, 256 * 4 * 4),  # Output: (256 * 4 * 4)
            nn.Unflatten(1, (256, 4, 4)),  # Output: (256, 4, 4)
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Output: (3, 32, 32)
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 84),  # Output: (84)
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(84, num_classes),  # Output: (num_classes)
        )

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