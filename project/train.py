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
        
        

def train_epoch(model, train_loader, criterion, optimizer, device, max_batches=None):
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)
    
    if max_batches is not None:
        num_batches = min(max_batches, num_batches)
    
    with tqdm(train_loader, desc="Training", total=num_batches, leave=False) as pbar:
        for batch_idx, data in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, _ = data  # For autoencoders, we use the inputs as targets
            inputs = inputs.to(device)  # Assuming you have a device defined
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
    
    return running_loss / num_batches


def validate(model, val_loader, criterion, device, max_batches=None):
    model.eval()
    val_loss = 0.0
    num_batches = len(val_loader)
    
    if max_batches is not None:
        num_batches = min(max_batches, num_batches)
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating", total=num_batches, leave=False) as pbar:
            for batch_idx, data in enumerate(pbar):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                inputs, _ = data
                inputs = inputs.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                
                val_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
    
    return val_loss / num_batches

def test_autoencoder(model, test_loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    num_batches = len(test_loader)
    images_losses = []

    if max_batches is not None:
        num_batches = min(max_batches, num_batches)

    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", total=num_batches, leave=False) as pbar:
            for batch_idx, data in enumerate(pbar):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                inputs, _ = data
                inputs = inputs.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                
                # Save images and their losses
                images_losses.append((inputs.cpu(), outputs.cpu(), loss.item()))
    
    avg_loss = total_loss / num_batches
    print(f'Test Loss: {avg_loss:.4f}')
    
    # Sort images by loss
    images_losses.sort(key=lambda x: x[2], reverse=True)
    
    # Create a directory to save the images
    os.makedirs('reconstructed_images', exist_ok=True)
    
    # Save the best images
    num_images_to_show = 5
    for i in range(num_images_to_show):
        original, reconstructed, loss = images_losses[i]
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(original[0].permute(1, 2, 0).numpy())
        axes[0].set_title('Original')
        axes[1].imshow(reconstructed[0].permute(1, 2, 0).numpy())
        axes[1].set_title('Reconstructed')
        plt.savefig(f'reconstructed_images/image_{i+1}.png')
        plt.close(fig)
    
    return avg_loss