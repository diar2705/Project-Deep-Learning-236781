import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from enum import Enum
import torchvision.transforms as transforms


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
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Adjusted T_max to match your training schedule
            eta_min=1e-6,  # Lower bound for the learning rate
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
        loss, accuracy = self._run_epoch(self.train_loader, Mode.TRAIN)
        self.scheduler.step()
        return loss, accuracy

    def validate(self):
        return self._run_epoch(self.val_loader, Mode.VAL)

    def test(self):
        test_loss, _ = self._run_epoch(self.test_loader, Mode.TEST)
        self._save_reconstructed_images(self.test_loader)
        return test_loss, 0.0  # Return dummy accuracy

    def _save_reconstructed_images(self, test_loader, num_images_to_show=5):
        self.model.eval()  # Set model to evaluation mode
        images_losses = []  # Store (original, reconstructed, loss) tuples

        # CIFAR-10 normalization values
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

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

        # Visualize and save the top `num_images_to_show` images
        for i in range(min(num_images_to_show, len(images_losses))):
            original, reconstructed, _ = images_losses[
                i
            ]  # Get original and reconstructed images
            fig, axes = plt.subplots(1, 2)  # Create a figure with two subplots

            # Reverse normalization
            original = (
                original * std + mean
            )  # Reverse normalization for original images
            reconstructed = (
                reconstructed * std + mean
            )  # Reverse normalization for reconstructed images

            # Clip pixel values to [0, 1]
            original = original.clamp(0, 1)
            reconstructed = reconstructed.clamp(0, 1)

            if original.shape[1] == 3:  # Color image (CIFAR-10)
                axes[0].imshow(
                    original[0].permute(1, 2, 0).numpy()
                )  # Display original image
                axes[1].imshow(
                    reconstructed[0].permute(1, 2, 0).numpy()
                )  # Display reconstructed image
            else:  # Grayscale image
                axes[0].imshow(original[0][0], cmap="gray")  # Display original image
                axes[1].imshow(
                    reconstructed[0][0], cmap="gray"
                )  # Display reconstructed image

            axes[0].set_title("Original")
            axes[1].set_title("Reconstructed")
            plt.savefig(f"reconstructed_images/image_{i+1}.png")  # Save the figure
            plt.close(fig)  # Close the figure to free memory


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

        # Improved scheduler with adjusted T_max and eta_min
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,  # Adjusted T_max to match your training schedule
            eta_min=1e-6,  # Lower bound for the learning rate
        )

    def train(self):
        loss, accuracy = self._run_epoch(self.train_loader, Mode.TRAIN)
        self.scheduler.step()
        return loss, accuracy

    def validate(self):
        return self._run_epoch(self.val_loader, Mode.VAL)

    def test(self):
        test_loss, accuracy = self._run_epoch(self.test_loader, Mode.TEST)
        return test_loss, accuracy


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # Concatenate positive pairs
        similarity_matrix = self.cosine_similarity(
            z.unsqueeze(1), z.unsqueeze(0)
        )  # Compute similarity matrix

        # Create positive mask
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        positive_mask = torch.cat([positive_mask, positive_mask], dim=0)
        positive_mask = torch.cat([positive_mask, positive_mask.T], dim=1)

        # Mask out self-similarities
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)

        # Compute logits
        logits = similarity_matrix / self.temperature
        logits = logits[mask].view(2 * batch_size, -1)

        # Compute positive logits
        positive_logits = similarity_matrix[positive_mask].view(2 * batch_size, -1)

        # Compute NT-Xent loss
        loss = -torch.log(
            torch.exp(positive_logits / self.temperature)
            / torch.exp(logits).sum(dim=1, keepdim=True)
        )
        return loss.mean()


class EnclassifierTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        super().__init__(model, train_loader, val_loader, test_loader, device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Adjusted T_max to match your training schedule
            eta_min=1e-6,  # Lower bound for the learning rate
        )

    def train(self):
        loss, accuracy = self._run_epoch(self.train_loader, Mode.TRAIN)
        self.scheduler.step()
        return loss, accuracy

    def validate(self):
        return self._run_epoch(self.val_loader, Mode.VAL)

    def test(self):
        test_loss, accuracy = self._run_epoch(self.test_loader, Mode.TEST)
        return test_loss, accuracy


class CLRTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        super(CLRTrainer, self).__init__(
            model, train_loader, val_loader, test_loader, device
        )
        self.criterion = NTXentLoss(temperature=0.3)
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Adjusted T_max to match your training schedule
            eta_min=1e-6,  # Lower bound for the learning rate
        )
        self.augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def train(self):
        loss, accuracy = self._run_epoch(self.train_loader, Mode.TRAIN)
        self.scheduler.step()
        return loss, accuracy

    def validate(self):
        return self._run_epoch(self.val_loader, Mode.VAL)

    def test(self):
        return self._run_epoch(self.test_loader, Mode.TEST)

    def _run_epoch(self, loader, mode: Mode):
        self.model.train() if mode == Mode.TRAIN else self.model.eval()
        total_loss = 0.0
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
            leave=True,
        ) as pbar:
            for inputs, _ in pbar:
                inputs = inputs.to(self.device)

                # Generate two augmented views of the same input
                inputs1 = self.augmentation(inputs)
                inputs2 = self.augmentation(inputs)

                if mode == Mode.TRAIN:
                    self.optimizer.zero_grad()

                # Forward pass
                z_i = self.model(inputs1)
                z_j = self.model(inputs2)

                # Compute loss
                loss = self.criterion(z_i, z_j)

                if mode == Mode.TRAIN:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        return total_loss / num_batches, 0.0  # Return dummy accuracy
