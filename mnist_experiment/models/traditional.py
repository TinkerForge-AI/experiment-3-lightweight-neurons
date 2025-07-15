"""
Traditional neural network architectures for MNIST classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TraditionalMLP(nn.Module):
    """
    Traditional Multi-Layer Perceptron for MNIST classification.
    """
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10, dropout=0.2):
        super(TraditionalMLP, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'Traditional MLP',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': 'Fully Connected'
        }


class TraditionalCNN(nn.Module):
    """
    Traditional Convolutional Neural Network for MNIST classification.
    """
    
    def __init__(self, num_classes=10, dropout=0.25):
        super(TraditionalCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3 (with padding)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'Traditional CNN',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': 'Convolutional'
        }


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet-like architecture."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class TraditionalResNet(nn.Module):
    """
    Simple ResNet-like architecture for MNIST.
    """
    
    def __init__(self, num_classes=10):
        super(TraditionalResNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual blocks
        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        
        # Global average pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'Traditional ResNet',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': 'Residual Convolutional'
        }


def get_traditional_model(model_type='cnn', **kwargs):
    """
    Factory function to get traditional models.
    
    Args:
        model_type (str): Type of model ('mlp', 'cnn', 'resnet')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        nn.Module: The requested model
    """
    if model_type.lower() == 'mlp':
        return TraditionalMLP(**kwargs)
    elif model_type.lower() == 'cnn':
        return TraditionalCNN(**kwargs)
    elif model_type.lower() == 'resnet':
        return TraditionalResNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    print("Testing traditional models...")
    
    # Test input
    x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
    
    models = {
        'MLP': TraditionalMLP(),
        'CNN': TraditionalCNN(),
        'ResNet': TraditionalResNet()
    }
    
    for name, model in models.items():
        try:
            output = model(x)
            info = model.get_model_info()
            print(f"\n{name}:")
            print(f"  Output shape: {output.shape}")
            print(f"  Parameters: {info['total_params']:,}")
            print(f"  Architecture: {info['architecture']}")
        except Exception as e:
            print(f"\n{name}: Error - {e}")
