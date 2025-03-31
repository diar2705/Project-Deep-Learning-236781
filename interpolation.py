import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from project.mnist import Encoder, Decoder

def interpolate_and_decode(decoder, z1, z2, n_steps=10):
    interpolated_images = []
    for a in torch.linspace(0, 1, n_steps):
        z = (1 - a) * z1 + a * z2  # Interpolate between z1 and z2
        decoded_image = decoder(z)  # Decode the interpolated latent vector
        interpolated_images.append(decoded_image)
    return interpolated_images

if __name__ == "__main__":
    # Load raw images (assuming already 28x28 grayscale)
    img1 = Image.open("/home/hadi.hbus/mini-project-236781/reconstructed_images/original_image_2.png")
    img2 = Image.open("/home/hadi.hbus/mini-project-236781/reconstructed_images/original_image_4.png")
    
    if img1.mode != 'L':
        img1 = img1.convert('L')
    if img2.mode != 'L':
        img2 = img2.convert('L')
    # Convert to tensors and normalize
    img1 = torch.from_numpy((np.array(img1,dtype=np.float32))/255.0 * 2 - 1).unsqueeze(0)  # [-1,1] range
    img2 = torch.from_numpy((np.array(img2,dtype=np.float32))/255.0 * 2 - 1).unsqueeze(0)  # [-1,1] range
    # Load models
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_state_dict(torch.load('encoder1.pth'))
    decoder.load_state_dict(torch.load('decoder1.pth'))
    
    print(img1.shape)
    # Process
    with torch.no_grad():
        z1 = encoder(img1.view(1, 1, 28, 28))
        z2 = encoder(img2.view(1, 1, 28, 28))
        
        # Decode interpolated latent vectors in a loop
        reconstructions = interpolate_and_decode(decoder, z1, z2, n_steps=10)
    
    # Visualize
    plt.figure(figsize=(15,3))
    for i, img in enumerate([img1] + reconstructions + [img2]):
        plt.subplot(1,12,i+1)
        plt.imshow(img.squeeze(), cmap='gray', vmin=-1, vmax=1)
        plt.axis('off')
    plt.savefig('interpolation.png')