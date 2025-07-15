"""
Training and evaluation utilities for MNIST experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import json
import os
from tqdm import tqdm
import psutil
import numpy as np


class ModelTrainer:
    """
    Unified trainer for both traditional and lightweight models.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': [],
            'memory_usage': []
        }
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, epoch_time
    
    def evaluate(self, test_loader, criterion):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            with tqdm(test_loader, desc='Evaluating') as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    total_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*correct/total:.2f}%'
                    })
        
        inference_time = time.time() - start_time
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, inference_time
    
    def measure_inference_speed(self, test_loader, num_batches=10):
        """Measure inference speed on a subset of data."""
        self.model.eval()
        
        times = []
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= num_batches:
                    break
                
                data = data.to(self.device)
                
                start_time = time.time()
                _ = self.model(data)
                end_time = time.time()
                
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return avg_time, std_time
    
    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            gpu_memory_cached = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
        else:
            gpu_memory = 0
            gpu_memory_cached = 0
        
        return {
            'cpu_memory_gb': memory_info.rss / 1024**3,
            'gpu_memory_gb': gpu_memory,
            'gpu_memory_cached_gb': gpu_memory_cached
        }
    
    def train(self, train_loader, test_loader, epochs=10, lr=0.001, 
              save_dir='./results', model_name='model'):
        """Complete training loop."""
        
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Tensorboard logging
        writer = SummaryWriter(os.path.join(save_dir, f'runs/{model_name}'))
        
        print(f"Training {model_name} for {epochs} epochs...")
        print(f"Device: {self.device}")
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Training
            train_loss, train_acc, epoch_time = self.train_epoch(train_loader, optimizer, criterion)
            
            # Evaluation
            val_loss, val_acc, _ = self.evaluate(test_loader, criterion)
            
            # Memory usage
            memory_usage = self.get_memory_usage()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['epoch_times'].append(epoch_time)
            self.training_history['memory_usage'].append(memory_usage)
            
            # Tensorboard logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            writer.add_scalar('Time/Epoch', epoch_time, epoch)
            writer.add_scalar('Memory/CPU_GB', memory_usage['cpu_memory_gb'], epoch)
            writer.add_scalar('Memory/GPU_GB', memory_usage['gpu_memory_gb'], epoch)
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Epoch Time: {epoch_time:.2f}s')
            print(f'Memory - CPU: {memory_usage["cpu_memory_gb"]:.2f}GB, '
                  f'GPU: {memory_usage["gpu_memory_gb"]:.2f}GB')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
                }, os.path.join(save_dir, f'{model_name}_best.pth'))
            
            # Step scheduler
            scheduler.step()
        
        writer.close()
        
        # Save training history
        with open(os.path.join(save_dir, f'{model_name}_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Final evaluation and timing
        print(f'\nFinal evaluation...')
        final_loss, final_acc, inference_time = self.evaluate(test_loader, criterion)
        avg_inference_time, std_inference_time = self.measure_inference_speed(test_loader)
        
        results = {
            'model_name': model_name,
            'final_accuracy': final_acc,
            'final_loss': final_loss,
            'best_accuracy': best_acc,
            'total_epochs': epochs,
            'avg_epoch_time': np.mean(self.training_history['epoch_times']),
            'total_training_time': sum(self.training_history['epoch_times']),
            'inference_time_full': inference_time,
            'avg_batch_inference_time': avg_inference_time,
            'std_batch_inference_time': std_inference_time,
            'final_memory_usage': memory_usage,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }
        
        # Save results
        with open(os.path.join(save_dir, f'{model_name}_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\nTraining completed!')
        print(f'Best Accuracy: {best_acc:.2f}%')
        print(f'Final Accuracy: {final_acc:.2f}%')
        print(f'Total Training Time: {sum(self.training_history["epoch_times"]):.2f}s')
        
        return results


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, 
                save_dir='./results', model_name='model'):
    """
    Convenience function to train a model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save results
        model_name: Name for saving files
        
    Returns:
        dict: Training results
    """
    trainer = ModelTrainer(model)
    return trainer.train(train_loader, test_loader, epochs, lr, save_dir, model_name)


if __name__ == "__main__":
    # Test the training module
    from models.traditional import TraditionalCNN
    from data.mnist_loader import get_mnist_dataloaders
    
    print("Testing training module...")
    
    # Load small dataset for testing
    train_loader, test_loader = get_mnist_dataloaders(batch_size=32)
    
    # Create a small model
    model = TraditionalCNN()
    
    # Test trainer
    trainer = ModelTrainer(model)
    print(f"Model loaded on device: {trainer.device}")
    
    # Test memory usage
    memory_usage = trainer.get_memory_usage()
    print(f"Memory usage: {memory_usage}")
    
    # Test inference speed
    avg_time, std_time = trainer.measure_inference_speed(test_loader, num_batches=5)
    print(f"Average inference time: {avg_time:.4f}s ± {std_time:.4f}s")
    
    print("Training module test completed!")
