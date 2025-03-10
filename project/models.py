import project.mnist as mn
import project.cifar10 as cf
import project.trainer as tr
import torch
import torch.nn as nn

def part1_self_supervised_autoencoder(model, train_loader, val_loader, test_loader, device, latent_dim=128):
    trainer = tr.AutoencoderTrainer(model, train_loader, val_loader, test_loader, device)
    trainer.fit()
    torch.save(model.encoder.state_dict(), "encoder.pth")

def part1_classifier(encoder, classifier, train_loader, val_loader, test_loader, device, latent_dim=128):
    encoder.load_state_dict(torch.load("encoder.pth"))
    for param in encoder.parameters():
        param.requires_grad = False
    model = nn.Sequential(encoder, classifier).to(device)
    trainer = tr.ClassifierTrainer(model, train_loader, val_loader, test_loader, device)
    trainer.fit()
    torch.save(model.state_dict(), "classifier_model.pth")
    
def aux(train_loader, val_loader, test_loader, device, mnist ,latent_dim=128):
    if mnist:
        model = mn.Autoencoder(latent_dim).to(device)
        part1_self_supervised_autoencoder(model, train_loader, val_loader, test_loader, device, latent_dim)
        encoder = mn.Encoder(latent_dim)
        classifier = mn.Classifier(num_classes=10)
        part1_classifier(encoder, classifier, train_loader, val_loader, test_loader, device, latent_dim)
    else:
        model = cf.Autoencoder(latent_dim).to(device)
        part1_self_supervised_autoencoder(model, train_loader, val_loader, test_loader, device, latent_dim)
        encoder = cf.Encoder(latent_dim)
        classifier = cf.Classifier(num_classes=10)
        part1_classifier(encoder, classifier, train_loader, val_loader, test_loader, device, latent_dim)