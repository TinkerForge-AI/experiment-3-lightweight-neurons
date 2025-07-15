"""
MNIST data loading and preprocessing utilities.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os


def get_mnist_dataloaders(batch_size=64, data_dir='./data', num_workers=2, pin_memory=True):
    """
    Load MNIST dataset and return train/test dataloaders.
    
    Args:
        batch_size (int): Batch size for dataloaders
        data_dir (str): Directory to store/load MNIST data
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_mnist_info():
    """
    Get information about the MNIST dataset.
    
    Returns:
        dict: Dataset information
    """
    return {
        'input_shape': (1, 28, 28),
        'num_classes': 10,
        'train_samples': 60000,
        'test_samples': 10000,
        'pixel_range': [0, 1],
        'mean': 0.1307,
        'std': 0.3081
    }


def visualize_samples(dataloader, num_samples=8):
    """
    Visualize sample images from the dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        num_samples (int): Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    # Get a batch of data
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Select samples to display
    samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(1, samples, figsize=(12, 3))
    if samples == 1:
        axes = [axes]
    
    for i in range(samples):
        img = images[i].squeeze()
        label = labels[i].item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test the data loading
    print("Testing MNIST data loading...")
    
    train_loader, test_loader = get_mnist_dataloaders(batch_size=32)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Display dataset info
    info = get_mnist_info()
    print("\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
