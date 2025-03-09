import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class Trainer():
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self, max_batches=None):
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        if max_batches is not None:
            num_batches = min(max_batches, num_batches)
        
        with tqdm(self.train_loader, desc="Training", total=num_batches, leave=False) as pbar:
            for batch_idx, data in enumerate(pbar):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                inputs, _ = data  # For autoencoders, we use the inputs as targets
                inputs = inputs.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        return running_loss / num_batches

    def validate(self, max_batches=None):
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        if max_batches is not None:
            num_batches = min(max_batches, num_batches)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validating", total=num_batches, leave=False) as pbar:
                for batch_idx, data in enumerate(pbar):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    inputs, _ = data
                    inputs = inputs.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, inputs)
                    
                    val_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
        
        return val_loss / num_batches

    def test(self, max_batches=None):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.test_loader)
        images_losses = []

        if max_batches is not None:
            num_batches = min(max_batches, num_batches)

        with torch.no_grad():
            with tqdm(self.test_loader, desc="Testing", total=num_batches, leave=False) as pbar:
                for batch_idx, data in enumerate(pbar):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    inputs, _ = data
                    inputs = inputs.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, inputs)

                    total_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())

                    images_losses.append((inputs.detach().cpu(), outputs.detach().cpu(), loss.item()))

        avg_loss = total_loss / max(1, num_batches)
        print(f'Test Loss: {avg_loss:.4f}')

        # Save top 5 highest-loss images
        top_losses = sorted(images_losses, key=lambda x: x[2], reverse=True)[:5]
        os.makedirs('reconstructed_images', exist_ok=True)

        for i, (original, reconstructed, loss) in enumerate(top_losses):
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(original[0].squeeze(0).numpy(), cmap='gray')
            axes[0].set_title('Original')
            axes[1].imshow(reconstructed[0].squeeze(0).numpy(), cmap='gray')
            axes[1].set_title('Reconstructed')
            plt.savefig(f'reconstructed_images/image_{i+1}.png')
            plt.close(fig)

        return avg_loss
