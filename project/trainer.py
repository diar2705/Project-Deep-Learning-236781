import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from enum import Enum
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn


class Mode(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def fit(self, num_epochs=1):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss, train_acc = self.train()
            val_loss, val_acc = self.validate()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            tqdm.write(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            print()
        test_loss, test_acc = self.test()
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n")
        self._plot_metrics(num_epochs)

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def _run_epoch(self, loader, mode: Mode):
        self.model.train() if mode == Mode.TRAIN else self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(loader)

        with tqdm(
            loader,
            desc=(
                "Training"
                if mode == Mode.TRAIN
                else (
                    "Validating"
                    if mode == Mode.VAL
                    else "Testing" if mode == Mode.TEST else "Unknown Mode"
                )
            ),
            total=num_batches,
            leave=True,
            dynamic_ncols=True,
            ncols=100,
        ) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if mode == Mode.TRAIN:
                    self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if mode == Mode.TRAIN:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                accuracy = 100 * correct / total
                pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

        return total_loss / num_batches, accuracy

    def _plot_metrics(self, num_epochs):
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label="Train Acc")
        plt.plot(epochs, self.val_accuracies, label="Val Acc")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig("accuracy_loss_plot.png")
        plt.close()


class AutoencoderTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        super().__init__(model, train_loader, val_loader, test_loader, device)
        self.criterion = torch.nn.L1Loss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # Reduce LR when the validation loss stops decreasing
            factor=0.5,  # Multiply LR by this factor when plateau is detected
            patience=5,  # Number of epochs with no improvement after which LR will be reduced
            min_lr=1e-6,  # Lower bound for the learning rate
        )

    def _run_epoch(self, loader, mode: Mode):
        self.model.train() if mode == Mode.TRAIN else self.model.eval()
        total_loss = 0.0
        with tqdm(
            loader,
            desc=(
                "Training"
                if mode == Mode.TRAIN
                else (
                    "Validating"
                    if mode == Mode.VAL
                    else "Testing" if mode == Mode.TEST else "Unknown Mode"
                )
            ),
            leave=True,
        ) as pbar:
            for inputs, _ in pbar:
                inputs = inputs.to(self.device)
                if mode == Mode.TRAIN:
                    self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                if mode == Mode.TRAIN:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        return total_loss / len(loader), 0.0  # Return dummy accuracy

    def train(self):
        return self._run_epoch(self.train_loader, Mode.TRAIN)
        

    def validate(self):
        val_loss, _ = self._run_epoch(self.val_loader, Mode.VAL)
        self.scheduler.step(val_loss)
        return val_loss, 0.0

    def test(self):
        test_loss, _ = self._run_epoch(self.test_loader, Mode.TEST)
        self._save_reconstructed_images(self.test_loader)
        return test_loss, 0.0  # Return dummy accuracy

    def _save_reconstructed_images(self, test_loader, num_images_to_show=5):
        self.model.eval()  # Set model to evaluation mode
        images_losses = []  # Store (original, reconstructed, loss) tuples

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)  # Move inputs to the device (GPU/CPU)
                outputs = self.model(inputs)  # Get reconstructed images
                loss = self.criterion(outputs, inputs).item()  # Compute loss
                images_losses.append(
                    (inputs.cpu(), outputs.cpu(), loss)
                )  # Store results on CPU

        # Sort images by loss (lowest loss first)
        images_losses.sort(key=lambda x: x[2])

        # Create directory for saving images
        os.makedirs("reconstructed_images", exist_ok=True)

        # Save the top `num_images_to_show` images
        for i in range(min(num_images_to_show, len(images_losses))):
            original, reconstructed, _ = images_losses[i]

            # Clip pixel values to [0, 1]
            original = original.clamp(0, 1)
            reconstructed = reconstructed.clamp(0, 1)

            # Save original image
            plt.imsave(
                f"reconstructed_images/original_image_{i+1}.png",
                original[0].permute(1, 2, 0).numpy() if original.shape[1] == 3 else original[0][0].numpy(),
                cmap="gray" if original.shape[1] != 3 else None
            )

            # Save reconstructed image
            plt.imsave(
                f"reconstructed_images/reconstructed_image_{i+1}.png",
                reconstructed[0].permute(1, 2, 0).numpy() if reconstructed.shape[1] == 3 else reconstructed[0][0].numpy(),
                cmap="gray" if reconstructed.shape[1] != 3 else None
            )


class ClassifierTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        super().__init__(model, train_loader, val_loader, test_loader, device)
        self.criterion = torch.nn.CrossEntropyLoss()
        # Lower weight decay
        self.optimizer = optim.AdamW(
            model[1].parameters(),
            lr=2e-3,  # Adjusted learning rate
            weight_decay=1e-4,  # Increased weight decay for better regularization
        )

        # Changed to ReduceLROnPlateau scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Reduce LR when the validation loss stops decreasing
            factor=0.5,  # Multiply LR by this factor when plateau is detected
            patience=5,  # Number of epochs with no improvement after which LR will be reduced
            min_lr=1e-6,  # Lower bound for the learning rate
        )

    def train(self):
        loss, accuracy = self._run_epoch(self.train_loader, Mode.TRAIN)
        # Note: with ReduceLROnPlateau, step should ideally be called after validation
        # with the validation loss as parameter
        return loss, accuracy

    def validate(self):
        val_loss, val_accuracy = self._run_epoch(self.val_loader, Mode.VAL)
        # Update scheduler based on validation loss
        self.scheduler.step(val_loss)
        return val_loss, val_accuracy

    def test(self):
        test_loss, accuracy = self._run_epoch(self.test_loader, Mode.TEST)
        return test_loss, accuracy


class EnclassifierTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        super().__init__(model, train_loader, val_loader, test_loader, device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # Changed to ReduceLROnPlateau scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Reduce LR when the validation loss stops decreasing
            factor=0.5,  # Multiply LR by this factor when plateau is detected
            patience=5,  # Number of epochs with no improvement after which LR will be reduced
            min_lr=1e-6,  # Lower bound for the learning rate
        )

    def train(self):
        return self._run_epoch(self.train_loader, Mode.TRAIN)

    def validate(self):
        val_loss, val_accuracy = self._run_epoch(self.val_loader, Mode.VAL)
        # Update scheduler based on validation loss
        self.scheduler.step(val_loss)
        return val_loss, val_accuracy

    def test(self):
        test_loss, accuracy = self._run_epoch(self.test_loader, Mode.TEST)
        return test_loss, accuracy


class NTXentLoss(nn.modules.loss._Loss):
    # NT-Xent loss for contrastive learning
    def __init__(self, temperature=0.5, reduction='mean'):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        device = z_i.device
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(similarity_matrix, batch_size)
        sim_j_i = torch.diag(similarity_matrix, -batch_size)
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0)
        mask = ~torch.eye(2 * batch_size, dtype=bool, device=device)
        negative_samples = similarity_matrix[mask].view(2 * batch_size, -1)
        logits = torch.cat([positive_samples.view(-1, 1), negative_samples], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        return loss

class CLRTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, is_mnist, device):
        super().__init__(model, train_loader, val_loader, test_loader, device)
        self.criterion = NTXentLoss(temperature=0.5)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-6,
        )
        self.is_mnist = is_mnist
        
    def train(self):
        loss, accuracy = self._run_epoch(self.train_loader, Mode.TRAIN)
        return loss, accuracy

    def validate(self):
        return self._run_epoch(self.val_loader, Mode.VAL)

    def test(self):
        return self._run_epoch(self.test_loader, Mode.TEST)

    def _run_epoch(self, loader, mode: Mode):
        self.model.train() if mode == Mode.TRAIN else self.model.eval()
        total_loss = 0.0
        num_batches = len(loader)
        device = self.device
        if self.is_mnist:
            augmentation = transforms.Compose([
                transforms.RandomRotation(15),  # Rotate images randomly
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Random translation
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),  # Add blur
            ])
        else:
            augmentation = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),  # More aggressive crop
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ])
        
        with tqdm(loader, desc=("Training" if mode == Mode.TRAIN else ("Validating" if mode == Mode.VAL else "Testing")), total=num_batches, leave=True, dynamic_ncols=True, ncols=100) as pbar:
            for inputs, _ in pbar:
                batch_size = inputs.size(0)
                if batch_size < 2:
                    continue
                inputs = inputs.to(device)
                if mode == Mode.TRAIN:
                    inputs1 = augmentation(inputs)
                    inputs2 = augmentation(inputs)
                else:
                    inputs1, inputs2 = inputs, inputs
                if mode == Mode.TRAIN:
                    self.optimizer.zero_grad()
                z_i = self.model(inputs1)
                z_j = self.model(inputs2)
                loss = self.criterion(z_i, z_j)
                if mode == Mode.TRAIN:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / num_batches, 0.0
