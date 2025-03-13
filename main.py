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
from project.models import aux

NUM_CLASSES = 10


def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", default=0, type=int, help="Seed for random number generators"
    )
    parser.add_argument(
        "--data-path",
        default="/datasets/cv_datasets/data",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument("--batch-size", default=8, type=int, help="Size of each batch")
    parser.add_argument(
        "--latent-dim", default=128, type=int, help="encoding dimension"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Default device to use",
    )
    parser.add_argument(
        "--mnist",
        action="store_true",
        default=False,
        help="Whether to use MNIST (True) or CIFAR10 (False) data",
    )
    parser.add_argument(
        "--self-supervised",
        action="store_true",
        default=False,
        help="Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)
    device = torch.device(args.device)
    print("Using device:", device)

    if args.mnist:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.1307], std=[0.3081]
                ),  # MNIST-specific normalization
            ]
        )
        train_dataset = datasets.MNIST(
            root=args.data_path, train=True, download=False, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=args.data_path, train=False, download=False, transform=transform
        )
    else:
        transform_train = transforms.Compose(
            [
                # Random horizontal flipping
                transforms.RandomHorizontalFlip(),
                # Random rotation (optional)
                transforms.RandomRotation(10),
                # Convert to tensor
                transforms.ToTensor(),
                # Normalize with mean and std for CIFAR-10
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],  # Normalize to [-1, 1] range
                    std=[0.5, 0.5, 0.5]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=transform_test
        )

    # When you create your dataloader you should split train_dataset or test_dataset to leave some aside for validation
    val_size = int(0.1 * len(train_dataset))  # 10% for validation
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    aux(train_loader, val_loader, test_loader, device, args.mnist, args.latent_dim, args.self_supervised)
