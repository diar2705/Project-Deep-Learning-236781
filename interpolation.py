import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def interpolate(z1, z2, n_steps=10):
    return torch.stack([(1-a)*z1 + a*z2 for a in torch.linspace(0, 1, n_steps)])

if __name__ == "__main__":
    # Load raw images (assuming already 28x28 grayscale)
    img1 = Image.open("./reconstructed_images/original_image_1.png")
    img2 = Image.open("./reconstructed_images/original_image_2.png")
    
    # Convert to tensors and normalize
    img1 = torch.FloatTensor(np.array(img1))/255.0 * 2 - 1  # [-1,1] range
    img2 = torch.FloatTensor(np.array(img2))/255.0 * 2 - 1
    
    # Load models
    encoder = torch.load("./encoder.pth")
    decoder = torch.load("./decoder.pth")
    
    # Process
    with torch.no_grad():
        z1 = encoder(img1.view(1, -1))
        z2 = encoder(img2.view(1, -1))
        reconstructions = decoder(interpolate(z1, z2))
    
    # Visualize
    plt.figure(figsize=(15,3))
    for i, img in enumerate([img1] + [reconstructions[i] for i in range(10)] + [img2]):
        plt.subplot(1,12,i+1)
        plt.imshow(img.squeeze(), cmap='gray', vmin=-1, vmax=1)
        plt.axis('off')
    plt.savefig('interpolation.png')