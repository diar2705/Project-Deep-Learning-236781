import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from enum import Enum
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
        
    def fit(self, num_epochs=1):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train()
            val_loss, val_acc = self.validate()
            tqdm.write(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            print("\n")
            test_loss, test_acc = self.test()
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n")

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def _run_epoch(self, loader, mode:Mode):
        self.model.train() if mode == Mode.TRAIN else self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(loader)

        with tqdm(
            loader,
            desc = (
                "Training" if mode == Mode.TRAIN else 
                "Validating" if mode == Mode.VAL else 
                "Testing" if mode == Mode.TEST else 
                "Unknown Mode"
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
                if isinstance(outputs, tuple):
                    outputs = outputs[1]
                loss = self.criterion(outputs, targets)

                if mode == Mode.TRAIN:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                accuracy = 100 * correct / total
                pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

        return total_loss / num_batches, accuracy


class AutoencoderTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        super().__init__(model, train_loader, val_loader, test_loader, device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(),lr=0.001)


    def _run_epoch(self, loader, mode:Mode):
        self.model.train() if mode == Mode.TRAIN else self.model.eval()
        total_loss = 0.0
        with tqdm(
            loader,
            desc = (
                "Training" if mode == Mode.TRAIN else 
                "Validating" if mode == Mode.VAL else 
                "Testing" if mode == Mode.TEST else 
                "Unknown Mode"
            )
            , leave=True) as pbar:
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
        return self._run_epoch(self.val_loader, Mode.VAL)

    def test(self):
        self.model.eval()
        test_loss, _ = self._run_epoch(self.test_loader, Mode.TEST)
        self._save_reconstructed_images(self.test_loader)
        return test_loss, 0.0  # Return dummy accuracy

    def _save_reconstructed_images(self, test_loader, num_images_to_show=5):
        self.model.eval()  # Set model to evaluation mode
        images_losses = []  # Store (original, reconstructed, loss) tuples

        # CIFAR-10 normalization values
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)  # Move inputs to the device (GPU/CPU)
                outputs = self.model(inputs)  # Get reconstructed images
                loss = self.criterion(outputs, inputs).item()  # Compute loss
                images_losses.append((inputs.cpu(), outputs.cpu(), loss))  # Store results on CPU

        # Sort images by loss (lowest loss first)
        images_losses.sort(key=lambda x: x[2])

        # Create directory for saving images
        os.makedirs("reconstructed_images", exist_ok=True)

        # Visualize and save the top `num_images_to_show` images
        for i in range(min(num_images_to_show, len(images_losses))):
            original, reconstructed, _ = images_losses[i]  # Get original and reconstructed images
            fig, axes = plt.subplots(1, 2)  # Create a figure with two subplots

            # Reverse normalization
            original = original * std + mean  # Reverse normalization for original images
            reconstructed = reconstructed * std + mean  # Reverse normalization for reconstructed images

            # Clip pixel values to [0, 1]
            original = original.clamp(0, 1)
            reconstructed = reconstructed.clamp(0, 1)

            if original.shape[1] == 3:  # Color image (CIFAR-10)
                axes[0].imshow(original[0].permute(1, 2, 0).numpy())  # Display original image
                axes[1].imshow(reconstructed[0].permute(1, 2, 0).numpy())  # Display reconstructed image
            else:  # Grayscale image
                axes[0].imshow(original[0][0], cmap="gray")  # Display original image
                axes[1].imshow(reconstructed[0][0], cmap="gray")  # Display reconstructed image

            axes[0].set_title("Original")
            axes[1].set_title("Reconstructed")
            plt.savefig(f"reconstructed_images/image_{i+1}.png")  # Save the figure
            plt.close(fig)  # Close the figure to free memory


class ClassifierTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        super().__init__(model, train_loader, val_loader, test_loader, device)
        self.criterion = torch.nn.CrossEntropyLoss()
        # Lower weight decay
        self.optimizer = optim.Adam(model[1].parameters(), lr=3e-4, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)

    def train(self):
        loss, accuracy = self._run_epoch(self.train_loader, Mode.TRAIN)
        self.scheduler.step()
        return loss, accuracy

    def validate(self):
        return self._run_epoch(self.val_loader, Mode.VAL)

    def test(self):
        self.model.eval()
        test_loss, accuracy = self._run_epoch(self.test_loader,Mode.TEST)
        return test_loss, accuracy

class EnclassifierTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        super().__init__(model, train_loader, val_loader, test_loader, device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=3e-4,weight_decay=0.05)
        
    def train(self):
        return self._run_epoch(self.train_loader, Mode.TRAIN)
    
    def validate(self):
        return self._run_epoch(self.val_loader, Mode.VAL)
    
    def test(self):
        self.model.eval()
        test_loss, accuracy = self._run_epoch(self.test_loader, Mode.TEST)
        return test_loss, accuracy