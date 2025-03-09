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
import project.autoencoder as ae
from project.train import train_autoencoder, validate_autoencoder, test_autoencoder,test_classifier,train_classifier,validate_classifier
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
    


def part1_self_supervised_autoencoder(train_loader,val_loader,test_loader,device,latent_dim=128):
    model = ae.Autoencoder(latent_dim).to(device)
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20
    train_losses = []
    val_losses  = []
    for epoch in range(num_epochs):
        train_loss = train_autoencoder(model, train_loader, criterion, optimizer, device)
        val_loss = validate_autoencoder(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        tqdm.write(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
    test_loss = test_autoencoder(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')
    
    # Save the model
    torch.save(model.encoder.state_dict(), 'encoder.pth')
    
def part1_classifier(train_loader,val_loader,test_loader,device,latent_dim=128):
    encoder = ae.Encoder()
    encoder.load_state_dict(torch.load('encoder.pth'))
    
    for param in encoder.parameters():
        param.requires_grad = False
    
    classifier = ae.Classifier(num_classes=10)

    # Combine encoder and classifier
    model = nn.Sequential(encoder, classifier).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 20
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_classifier(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_classifier(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        tqdm.write(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Test the model
    test_loss, test_acc = test_classifier(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    # Save the model
    torch.save(model.state_dict(), 'classifier_model.pth')
    
    return model, test_acc
    
    
    
    

    

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
    
    
    part1_self_supervised_autoencoder(train_loader,val_loader,test_loader,device)
    part1_classifier(train_loader,val_loader,test_loader,device)
    

    
    
    
    
    
