import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def train_autoencoder(model, train_loader, criterion, optimizer, device):
    """
    Train an autoencoder model for one epoch.
    
    Args:
        model: The autoencoder model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run the model on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)

    with tqdm(train_loader, desc="Training", total=num_batches, leave=True, dynamic_ncols=True, ncols=100) as pbar:
        for data in pbar:
            inputs, _ = data  # For autoencoders, we use the inputs as targets
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            # Removed redundant pbar.update(1)

    return running_loss / num_batches


def validate_autoencoder(model, val_loader, criterion, device):
    """
    Validate an autoencoder model.
    
    Args:
        model: The autoencoder model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run the model on
        
    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        with tqdm(val_loader, desc="Validating", total=num_batches, leave=True, dynamic_ncols=True, ncols=100) as pbar:
            for data in pbar:
                inputs, _ = data
                inputs = inputs.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, inputs)

                val_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                # Removed redundant pbar.update(1)

    return val_loss / num_batches


def test_autoencoder(model, test_loader, criterion, device):
    """
    Test an autoencoder model and save reconstructed images.
    
    Args:
        model: The autoencoder model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run the model on
        
    Returns:
        Average test loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(test_loader)
    images_losses = []

    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", total=num_batches, leave=True, dynamic_ncols=True, ncols=100) as pbar:
            for data in pbar:  # Removed unnecessary batch_idx
                inputs, _ = data
                inputs = inputs.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                # Removed redundant pbar.update(1)

                # Save images and their losses
                images_losses.append((inputs.cpu(), outputs.cpu(), loss.item()))

    avg_loss = total_loss / num_batches

    # Sort images by loss
    images_losses.sort(key=lambda x: x[2])

    # Create a directory to save the images
    os.makedirs('reconstructed_images', exist_ok=True)

    # Save the best images
    num_images_to_show = 5
    for i in range(min(num_images_to_show, len(images_losses))):  # Added check to avoid index error
        original, reconstructed, loss = images_losses[i]
        fig, axes = plt.subplots(1, 2)
        
        # Handle different channel configurations safely
        original_img = original[0]
        reconstructed_img = reconstructed[0]
        
        # Check if image has 3 channels (RGB)
        if original_img.shape[0] == 3:
            axes[0].imshow(original_img.permute(1, 2, 0).numpy())
            axes[1].imshow(reconstructed_img.permute(1, 2, 0).numpy())
        else:  # Assume grayscale
            axes[0].imshow(original_img[0], cmap='gray')
            axes[1].imshow(reconstructed_img[0], cmap='gray')
            
        axes[0].set_title('Original')
        axes[1].set_title('Reconstructed')
        plt.savefig(f'reconstructed_images/image_{i+1}.png')
        plt.close(fig)

    return avg_loss


def train_classifier(model, train_loader, optimizer, criterion, device):
    """
    Train a classifier model for one epoch.
    
    Args:
        model: The classifier model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to run the model on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    num_batches = len(train_loader)
    
    with tqdm(train_loader, desc="Training", total=num_batches, leave=True, dynamic_ncols=True, ncols=100) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[1], targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _,predicted = outputs[1].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar with current loss and accuracy
            current_accuracy = 100 * correct / total
            pbar.set_postfix(loss=loss.item(), accuracy=f"{current_accuracy:.2f}%")
            # Removed redundant pbar.update(1)
    
    accuracy = 100 * correct / total
    avg_loss = train_loss / num_batches
    
    return avg_loss, accuracy
    

def test_classifier(model, test_loader, criterion, device):
    """
    Test a classifier model.
    
    Args:
        model: The classifier model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run the model on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_batches = len(test_loader)
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", total=num_batches, leave=True, dynamic_ncols=True, ncols=100) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs[1], targets)
                
                test_loss += loss.item()
                _,predicted = outputs[1].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar with current loss and accuracy
                current_accuracy = 100 * correct / total
                pbar.set_postfix(loss=loss.item(), accuracy=f"{current_accuracy:.2f}%")
                # Removed redundant pbar.update(1)
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / num_batches
    
    return avg_loss, accuracy


def validate_classifier(model, val_loader, criterion, device):
    """
    Validate a classifier model.
    
    Args:
        model: The classifier model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run the model on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating", total=num_batches, leave=True, dynamic_ncols=True, ncols=100) as pbar:
            for data in pbar:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs[1], labels)
                
                val_loss += loss.item()
                
                _,predicted = torch.max(outputs[1], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
                pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")
                # Removed redundant pbar.update(1)
    
    return val_loss / num_batches, accuracy
