#!/usr/bin/env python3
"""
Test script to verify the MNIST experiment setup.
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test MNIST data loading."""
    print("Testing MNIST data loading...")
    try:
        from data.mnist_loader import get_mnist_dataloaders, get_mnist_info
        
        train_loader, test_loader = get_mnist_dataloaders(batch_size=32)
        print(f"✓ Data loaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test a batch
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Test dataset info
        info = get_mnist_info()
        print(f"  Dataset info: {info}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_traditional_models():
    """Test traditional model creation."""
    print("\nTesting traditional models...")
    try:
        from models.traditional import get_traditional_model
        
        test_input = torch.randn(8, 1, 28, 28)
        
        models = ['mlp', 'cnn', 'resnet']
        for model_type in models:
            model = get_traditional_model(model_type)
            output = model(test_input)
            info = model.get_model_info()
            print(f"  ✓ {model_type.upper()}: {output.shape}, params: {info['total_params']:,}")
        
        return True
    except Exception as e:
        print(f"✗ Traditional models failed: {e}")
        return False


def test_lightweight_models():
    """Test lightweight model creation."""
    print("\nTesting lightweight models...")
    try:
        from models.lightweight import get_lightweight_model
        
        test_input = torch.randn(8, 1, 28, 28)
        
        models = ['basic', 'conv', 'adaptive']
        for model_type in models:
            model = get_lightweight_model(model_type)
            output = model(test_input)
            info = model.get_model_info()
            print(f"  ✓ {model_type.upper()}: {output.shape}, params: {info['total_params']:,}")
        
        return True
    except Exception as e:
        print(f"✗ Lightweight models failed: {e}")
        return False


def test_training():
    """Test training functionality."""
    print("\nTesting training functionality...")
    try:
        from training.train import ModelTrainer
        from models.traditional import TraditionalCNN
        from data.mnist_loader import get_mnist_dataloaders
        
        # Create small dataset
        train_loader, test_loader = get_mnist_dataloaders(batch_size=32)
        
        # Create model and trainer
        model = TraditionalCNN()
        trainer = ModelTrainer(model)
        
        # Test memory usage
        memory_usage = trainer.get_memory_usage()
        print(f"  ✓ Memory usage: {memory_usage}")
        
        # Test inference speed
        avg_time, std_time = trainer.measure_inference_speed(test_loader, num_batches=3)
        print(f"  ✓ Inference speed: {avg_time:.4f}s ± {std_time:.4f}s")
        
        return True
    except Exception as e:
        print(f"✗ Training functionality failed: {e}")
        return False


def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics functionality...")
    try:
        from utils.metrics import calculate_efficiency_metrics, compare_models
        
        # Sample results
        sample_results = [
            {
                'model_name': 'Test Model 1',
                'final_accuracy': 95.0,
                'best_accuracy': 95.5,
                'total_training_time': 100.0,
                'avg_epoch_time': 10.0,
                'avg_batch_inference_time': 0.01,
                'final_memory_usage': {'cpu_memory_gb': 1.0, 'gpu_memory_gb': 0.5},
                'model_info': {'total_params': 10000, 'architecture': 'Test Architecture'}
            }
        ]
        
        # Test efficiency metrics
        efficiency = calculate_efficiency_metrics(sample_results[0])
        print(f"  ✓ Efficiency metrics calculated: {len(efficiency)} metrics")
        
        # Test comparison
        comparison_df = compare_models(sample_results)
        print(f"  ✓ Comparison DataFrame: {comparison_df.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics functionality failed: {e}")
        return False


def main():
    """Run all tests."""
    print("MNIST Experiment Setup Test")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_traditional_models,
        test_lightweight_models,
        test_training,
        test_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The experiment setup is ready.")
        print("\nTo run the full experiment:")
        print("  python experiments/compare.py")
        print("\nTo run a quick test:")
        print("  python experiments/compare.py --mode quick")
    else:
        print("✗ Some tests failed. Please check the setup.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
