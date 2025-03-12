import project.mnist as mn
import project.cifar10 as cf
import project.trainer as tr
import torch
import torch.nn as nn


def Part1(
    model,
    encoder,
    classifier,
    train_loader,
    val_loader,
    test_loader,
    device,
    latent_dim=128,
):
    trainer = tr.AutoencoderTrainer(
        model, train_loader, val_loader, test_loader, device
    )
    trainer.fit(30)
    torch.save(model.encoder.state_dict(), "encoder.pth")
    encoder.load_state_dict(torch.load("encoder.pth"))
    for param in encoder.parameters():
        param.requires_grad = False
    model = nn.Sequential(encoder, classifier).to(device)
    trainer = tr.ClassifierTrainer(model, train_loader, val_loader, test_loader, device)
    trainer.fit(50)
    torch.save(model.state_dict(), "classifier_model.pth")

def Part2(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    latent_dim=128,
):
    trainer = tr.EnclassifierTrainer(
        model, train_loader, val_loader, test_loader, device
    )
    trainer.fit(20)


def aux(train_loader, val_loader, test_loader, device, mnist, latent_dim=128):
    if mnist:
        model = mn.Autoencoder(latent_dim).to(device)
        encoder = mn.Encoder(latent_dim)
        classifier = mn.Classifier(num_classes=10)
        model = mn.Enclassifier(latent_dim).to(device)
    else:
        model = cf.Autoencoder(latent_dim).to(device)
        encoder = cf.Encoder(latent_dim)
        classifier = cf.Classifier(num_classes=10)

    Part1(
        model,
        encoder,
        classifier,
        train_loader,
        val_loader,
        test_loader,
        device,
        latent_dim,
    )
