"""
Lightweight neural network architectures using single-weight neurons and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleWeightNeuron(nn.Module):
    """
    A single-weight neuron that learns a single scalar weight per input feature.
    """
    
    def __init__(self, input_size):
        super(SingleWeightNeuron, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Element-wise multiplication and sum
        return torch.sum(x * self.weight, dim=-1, keepdim=True) + self.bias


class AttentionModule(nn.Module):
    """
    Simple attention mechanism for feature weighting.
    """
    
    def __init__(self, input_size, attention_size=64):
        super(AttentionModule, self).__init__()
        self.attention_size = attention_size
        
        # Attention layers
        self.query = nn.Linear(input_size, attention_size)
        self.key = nn.Linear(input_size, attention_size)
        self.value = nn.Linear(input_size, attention_size)
        
        # Output projection
        self.output_proj = nn.Linear(attention_size, input_size)
        
        # Scale factor
        self.scale = math.sqrt(attention_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Reshape for attention computation
        x_reshaped = x.view(batch_size, 1, seq_len)  # Treat features as sequence
        
        # Compute Q, K, V
        Q = self.query(x_reshaped)  # (batch, 1, attention_size)
        K = self.key(x_reshaped)    # (batch, 1, attention_size)
        V = self.value(x_reshaped)  # (batch, 1, attention_size)
        
        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)  # (batch, 1, attention_size)
        
        # Project back to input size
        output = self.output_proj(attended.squeeze(1))  # (batch, input_size)
        
        return output


class LightweightNeuronLayer(nn.Module):
    """
    A layer of lightweight single-weight neurons with optional attention.
    """
    
    def __init__(self, input_size, num_neurons, use_attention=True, attention_size=64):
        super(LightweightNeuronLayer, self).__init__()
        
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.use_attention = use_attention
        
        # Create multiple single-weight neurons
        self.neurons = nn.ModuleList([
            SingleWeightNeuron(input_size) for _ in range(num_neurons)
        ])
        
        # Optional attention mechanism
        if use_attention:
            self.attention = AttentionModule(input_size, attention_size)
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x) + x  # Residual connection
        
        # Apply each neuron
        neuron_outputs = []
        for neuron in self.neurons:
            output = neuron(x)
            neuron_outputs.append(output)
        
        # Concatenate outputs
        output = torch.cat(neuron_outputs, dim=-1)  # (batch, num_neurons)
        
        return self.activation(output)


class LightweightNetwork(nn.Module):
    """
    Complete lightweight network using single-weight neurons and attention.
    """
    
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, 
                 use_attention=True, attention_size=64, dropout=0.1):
        super(LightweightNetwork, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Build lightweight layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(
                LightweightNeuronLayer(
                    prev_size, hidden_size, use_attention, attention_size
                )
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer (traditional linear layer for classification)
        self.output_layer = nn.Linear(prev_size, num_classes)
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Pass through lightweight layers
        x = self.hidden_layers(x)
        
        # Final classification
        x = self.output_layer(x)
        
        return x
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'Lightweight Network',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': 'Single-Weight Neurons + Attention'
        }


class LightweightConvNetwork(nn.Module):
    """
    Lightweight convolutional network with attention-based feature extraction.
    """
    
    def __init__(self, num_classes=10, use_attention=True, dropout=0.1):
        super(LightweightConvNetwork, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Lightweight feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # Larger kernel, fewer channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Attention for spatial features
        if use_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # Lightweight classifier
        feature_size = 32 * 7 * 7  # After two pooling operations: 28->14->7
        self.classifier = LightweightNeuronLayer(
            feature_size, 64, use_attention=use_attention, attention_size=32
        )
        
        self.output_layer = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Lightweight convolution
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Spatial attention
        if self.use_attention:
            attention_map = self.spatial_attention(x)
            x = x * attention_map
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        
        return x
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'Lightweight Conv Network',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': 'Lightweight Conv + Single-Weight Neurons + Attention'
        }


class AdaptiveLightweightNetwork(nn.Module):
    """
    Adaptive lightweight network that learns to route information efficiently.
    """
    
    def __init__(self, input_size=784, num_classes=10, num_experts=4, expert_size=32):
        super(AdaptiveLightweightNetwork, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_experts = num_experts
        
        # Router network (decides which experts to use)
        self.router = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert networks (lightweight single-weight neuron layers)
        self.experts = nn.ModuleList([
            LightweightNeuronLayer(input_size, expert_size, use_attention=True)
            for _ in range(num_experts)
        ])
        
        # Final classifier
        self.classifier = nn.Linear(expert_size, num_classes)
        
    def forward(self, x):
        # Flatten input
        x_flat = x.view(x.size(0), -1)
        
        # Get routing weights
        routing_weights = self.router(x_flat)  # (batch, num_experts)
        
        # Apply experts and combine
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x_flat)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, expert_size)
        
        # Weighted combination
        routing_weights = routing_weights.unsqueeze(-1)  # (batch, num_experts, 1)
        combined_output = torch.sum(expert_outputs * routing_weights, dim=1)  # (batch, expert_size)
        
        # Final classification
        output = self.classifier(combined_output)
        
        return output
    
    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'Adaptive Lightweight Network',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': 'Mixture of Lightweight Experts + Routing'
        }


def get_lightweight_model(model_type='basic', **kwargs):
    """
    Factory function to get lightweight models.
    
    Args:
        model_type (str): Type of model ('basic', 'conv', 'adaptive')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        nn.Module: The requested lightweight model
    """
    if model_type.lower() == 'basic':
        return LightweightNetwork(**kwargs)
    elif model_type.lower() == 'conv':
        return LightweightConvNetwork(**kwargs)
    elif model_type.lower() == 'adaptive':
        return AdaptiveLightweightNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown lightweight model type: {model_type}")


# Aliases for import (must be before __main__ block)
class NoAttentionLightweightNetwork(LightweightNetwork):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, dropout=0.1):
        super().__init__(input_size, hidden_sizes, num_classes, use_attention=False, dropout=dropout)

Basic = LightweightNetwork
Conv = LightweightConvNetwork
Adaptive = AdaptiveLightweightNetwork
NoAttention = NoAttentionLightweightNetwork

__all__ = [
    "Basic", "Conv", "Adaptive", "NoAttention",
    "LightweightNetwork", "LightweightConvNetwork", "AdaptiveLightweightNetwork", "NoAttentionLightweightNetwork",
    "SingleWeightNeuron", "AttentionModule", "LightweightNeuronLayer",
    "get_lightweight_model"
]


if __name__ == "__main__":
    # Test the lightweight models
    print("Testing lightweight models...")
    
    # Test input
    x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
    
    models = {
        'Lightweight Basic': LightweightNetwork(),
        'Lightweight Conv': LightweightConvNetwork(),
        'Adaptive Lightweight': AdaptiveLightweightNetwork()
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
    
    # Test individual components
    print("\nTesting individual components...")
    
    # Test SingleWeightNeuron
    swn = SingleWeightNeuron(784)
    x_flat = torch.randn(32, 784)
    swn_output = swn(x_flat)
    print(f"SingleWeightNeuron output shape: {swn_output.shape}")
    
    # Test AttentionModule
    attention = AttentionModule(784, 64)
    att_output = attention(x_flat)
    print(f"AttentionModule output shape: {att_output.shape}")
