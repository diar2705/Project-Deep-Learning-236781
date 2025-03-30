import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import argparse
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
        "--part",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Specify which part of the project to run (1, 2, or 3).",
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
                transforms.Normalize(mean=[0.1307], std=[0.3081]),
            ]
        )
        train_dataset = datasets.MNIST(
            root=args.data_path, train=True, download=False, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=args.data_path, train=False, download=False, transform=transform
        )
        full_train_dataset = train_dataset
    else:
        # Define separate transforms for training and testing/validation
        if args.part != 3:
            transform_train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505 ,0.26158768]),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505 ,0.26158768]),
                ]
            )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505 ,0.26158768]),
            ]
        )

        # Create two separate dataset instances: one for training (with augmentations) and one for validation (without augmentations)
        full_train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=transform_train
        )
        full_val_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=transform_test
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=transform_test
        )

    # Define the split size and obtain indices
    val_size = int(0.1 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    indices = list(range(len(full_train_dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subsets with corresponding transforms
    if args.mnist:
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)
    else:
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    aux(
        train_loader,
        val_loader,
        test_loader,
        device,
        args.mnist,
        args.latent_dim,
        args.part,
    )
