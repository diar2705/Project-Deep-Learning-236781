import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import project.mnist.models as ae
from project.mnist.trainer import AutoencoderTrainer, ClassifierTrainer
from tqdm import tqdm

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    return parser.parse_args()
    


def part1_self_supervised_autoencoder(train_loader, val_loader, test_loader, device, latent_dim=128):
    model = ae.Autoencoder(latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = AutoencoderTrainer(model, train_loader, val_loader, test_loader, device, criterion, optimizer)
    trainer.fit()
    torch.save(model.encoder.state_dict(), "encoder.pth")

def part1_classifier(train_loader, val_loader, test_loader, device, latent_dim=128):
    encoder = ae.Encoder(latent_dim)
    encoder.load_state_dict(torch.load("encoder.pth"))
    for param in encoder.parameters():
        param.requires_grad = False
    classifier = ae.Classifier(num_classes=10)
    model = nn.Sequential(encoder, classifier).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    trainer = ClassifierTrainer(model, train_loader, val_loader, test_loader, device, criterion, optimizer)
    trainer.fit()
    torch.save(model.state_dict(), "classifier_model.pth")

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  #one possible convenient normalization. You don't have to use it.
    ])
    args = get_args()
    freeze_seeds(args.seed)
    device = torch.device(args.device)
    print("Using device:", device)
                                           
    if args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])  # MNIST-specific normalization
        ])
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        
    # When you create your dataloader you should split train_dataset or test_dataset to leave some aside for validation
    val_size = int(0.1 * len(train_dataset))  # 10% for validation
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    part1_self_supervised_autoencoder(train_loader, val_loader, test_loader, device, args.latent_dim)
    part1_classifier(train_loader, val_loader, test_loader, device, args.latent_dim)

