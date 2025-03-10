import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.optim as optim


class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    def _run_epoch(self, loader, train_mode=True):
        self.model.train() if train_mode else self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(loader)

        with tqdm(
            loader,
            desc="Training" if train_mode else "Validating",
            total=num_batches,
            leave=True,
            dynamic_ncols=True,
            ncols=100,
        ) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if train_mode:
                    self.optimizer.zero_grad()

                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[1]
                loss = self.criterion(outputs, targets)

                if train_mode:
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

    def _run_epoch(self, loader, train_mode=True):
        self.model.train() if train_mode else self.model.eval()
        total_loss = 0.0
        with tqdm(
            loader, desc="Training" if train_mode else "Validating", leave=True
        ) as pbar:
            for inputs, _ in pbar:
                inputs = inputs.to(self.device)
                if train_mode:
                    self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                if train_mode:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        return total_loss / len(loader), 0.0  # Return dummy accuracy

    def train(self):
        return self._run_epoch(self.train_loader, train_mode=True)

    def validate(self):
        return self._run_epoch(self.val_loader, train_mode=False)

    def test(self):
        self.model.eval()
        test_loss, _ = self._run_epoch(self.test_loader, train_mode=False)
        self._save_reconstructed_images(self.test_loader)
        return test_loss, 0.0  # Return dummy accuracy

    def _save_reconstructed_images(self, test_loader, num_images_to_show=5):
        self.model.eval()
        images_losses = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs).item()
                images_losses.append((inputs.cpu(), outputs.cpu(), loss))

        images_losses.sort(key=lambda x: x[2])
        os.makedirs("reconstructed_images", exist_ok=True)

        for i in range(min(num_images_to_show, len(images_losses))):
            original, reconstructed, _ = images_losses[i]
            fig, axes = plt.subplots(1, 2)

            if original.shape[1] == 3:
                axes[0].imshow(original[0].permute(1, 2, 0).numpy())
                axes[1].imshow(reconstructed[0].permute(1, 2, 0).numpy())
            else:
                axes[0].imshow(original[0][0], cmap="gray")
                axes[1].imshow(reconstructed[0][0], cmap="gray")

            axes[0].set_title("Original")
            axes[1].set_title("Reconstructed")
            plt.savefig(f"reconstructed_images/image_{i+1}.png")
            plt.close(fig)


class ClassifierTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        super().__init__(model, train_loader, val_loader, test_loader, device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        return self._run_epoch(self.train_loader, train_mode=True)

    def validate(self):
        return self._run_epoch(self.val_loader, train_mode=False)

    def test(self):
        self.model.eval()
        test_loss, accuracy = self._run_epoch(self.test_loader, train_mode=False)
        return test_loss, accuracy
