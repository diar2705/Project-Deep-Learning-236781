import project.mnist as mn
import project.cifar10 as cf
import project.trainer as tr
import torch
import torch.nn as nn
from enum import Enum
from utils import plot_tsne


class Part(Enum):
    ONE = 1
    TWO = 2


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class Enclassifier(nn.Module):
    def __init__(self, encoder, classifier, latent_dim=128):
        super(Enclassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)


def Part1(
    model,
    encoder,
    classifier,
    train_loader,
    val_loader,
    test_loader,
    device,
    is_mnist,
    latent_dim=128,
):
    trainer = tr.AutoencoderTrainer(
        model, train_loader, val_loader, test_loader, device
    )
    if is_mnist:
        trainer.fit(10)
    else:
        trainer.fit(30)
    torch.save(model.encoder.state_dict(), "encoder1.pth")
    encoder.load_state_dict(torch.load("encoder1.pth"))
    for param in encoder.parameters():
        param.requires_grad = False
    model = nn.Sequential(encoder, classifier).to(device)
    trainer = tr.ClassifierTrainer(model, train_loader, val_loader, test_loader, device)
    if is_mnist:
        trainer.fit(10)
    else:
        trainer.fit(50)
    torch.save(model.state_dict(), "classifier_model1.pth")


def Part3(
    encoder,
    classifier,
    train_loader,
    val_loader,
    test_loader,
    device,
    latent_dim=128,
):
    encoder = encoder.to(device)
    trainer = tr.CLRTrainer(encoder, train_loader, val_loader, test_loader, device)
    trainer.fit(30)
    torch.save(encoder.state_dict(), "encoder3.pth")
    encoder.load_state_dict(torch.load("encoder3.pth"))
    for param in encoder.parameters():
        param.requires_grad = False
    model = nn.Sequential(encoder, classifier).to(device)
    trainer = tr.ClassifierTrainer(model, train_loader, val_loader, test_loader, device)
    trainer.fit(25)
    torch.save(model.state_dict(), "classifier_model3.pth")


def aux(train_loader, val_loader, test_loader, device, is_mnist, latent_dim, part):
    if is_mnist:
        encoder = mn.Encoder(latent_dim)
        decoder = mn.Decoder(latent_dim)
        classifier = mn.Classifier(num_classes=10)
    else:
        encoder = cf.Encoder(latent_dim)
        decoder = cf.Decoder(latent_dim)
        classifier = cf.Classifier(num_classes=10)

    final_encoder = None

    if part == 1:
        model = Autoencoder(encoder, decoder, latent_dim).to(device)
        Part1(
            model,
            encoder,
            classifier,
            train_loader,
            val_loader,
            test_loader,
            device,
            is_mnist,
            latent_dim,
        )
        final_encoder = encoder
    elif part == 2:
        model = Enclassifier(encoder, classifier, latent_dim).to(device)
        trainer = tr.EnclassifierTrainer(
            model, train_loader, val_loader, test_loader, device
        )
        trainer.fit(20)
        final_encoder = model.encoder
    elif part == 3:
        Part3(
            encoder,
            classifier,
            train_loader,
            val_loader,
            test_loader,
            device,
            latent_dim,
        )
        final_encoder = encoder
    plot_tsne(final_encoder, test_loader, device)
