"""
Main comparison script for MNIST neural network architecture experiment.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mnist_loader import get_mnist_dataloaders
from models.traditional import get_traditional_model
from models.lightweight import get_lightweight_model
from training.train import train_model
from utils.metrics import compare_models, rank_models, generate_summary_report
from utils.plotting import create_summary_dashboard


def setup_experiment(config):
    """Setup experiment configuration and create results directory."""
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(config['base_results_dir'], f"experiment_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save experiment configuration
    import json
    config_path = os.path.join(results_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment results will be saved to: {results_dir}")
    return results_dir


def run_single_experiment(model_config, train_loader, test_loader, results_dir, device):
    """Run experiment for a single model configuration."""
    
    model_name = model_config['name']
    model_type = model_config['type']
    model_category = model_config['category']
    model_params = model_config.get('params', {})
    training_params = model_config.get('training', {})
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"Type: {model_type}, Category: {model_category}")
    print(f"Parameters: {model_params}")
    print(f"{'='*60}")
    
    try:
        # Create model
        if model_category == 'traditional':
            model = get_traditional_model(model_type, **model_params)
        elif model_category == 'lightweight':
            model = get_lightweight_model(model_type, **model_params)
        else:
            raise ValueError(f"Unknown model category: {model_category}")
        
        # Print model information
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            print(f"Model Info: {info}")
        
        # Train model
        results = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=training_params.get('epochs', 10),
            lr=training_params.get('lr', 0.001),
            save_dir=results_dir,
            model_name=model_name
        )
        
        print(f"Training completed for {model_name}")
        print(f"Final Accuracy: {results['final_accuracy']:.2f}%")
        print(f"Training Time: {results['total_training_time']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return None


def run_comparison_experiment(config):
    """Run the complete comparison experiment."""
    
    print("Starting MNIST Neural Network Architecture Comparison Experiment")
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Setup experiment
    results_dir = setup_experiment(config)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Run experiments for each model
    all_results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_config in config['models']:
        if model_config.get('enabled', True):
            results = run_single_experiment(
                model_config, train_loader, test_loader, results_dir, device
            )
            if results:
                all_results.append(results)
    
    # Generate comparison and analysis
    if len(all_results) >= 2:
        print(f"\n{'='*60}")
        print("GENERATING COMPARISON ANALYSIS")
        print(f"{'='*60}")
        
        # Create comparison dataframe
        comparison_df = compare_models(all_results)
        ranked_df = rank_models(comparison_df)
        
        # Save comparison results
        comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
        ranked_df.to_csv(os.path.join(results_dir, 'model_ranking.csv'), index=False)
        
        # Generate summary report
        summary_report = generate_summary_report(results_dir, 
                                                os.path.join(results_dir, 'summary_report.txt'))
        print("\n" + summary_report)
        
        # Create visualizations
        print("\nGenerating visualization dashboard...")
        plot_dir = os.path.join(results_dir, 'plots')
        create_summary_dashboard(results_dir, plot_dir)
        
        print(f"\nExperiment completed! Results saved to: {results_dir}")
        
    else:
        print(f"\nInsufficient results for comparison (got {len(all_results)} models)")
    
    return results_dir, all_results


def get_default_config():
    """Get default experiment configuration."""
    
    return {
        'base_results_dir': './results',
        'data': {
            'batch_size': 64,
            'num_workers': 2
        },
        'models': [
            # Traditional Models
            {
                'name': 'Traditional_MLP',
                'category': 'traditional',
                'type': 'mlp',
                'enabled': True,
                'params': {
                    'hidden_sizes': [512, 256],
                    'dropout': 0.2
                },
                'training': {
                    'epochs': 15,
                    'lr': 0.001
                }
            },
            {
                'name': 'Traditional_CNN',
                'category': 'traditional',
                'type': 'cnn',
                'enabled': True,
                'params': {
                    'dropout': 0.25
                },
                'training': {
                    'epochs': 15,
                    'lr': 0.001
                }
            },
            {
                'name': 'Traditional_ResNet',
                'category': 'traditional',
                'type': 'resnet',
                'enabled': True,
                'params': {},
                'training': {
                    'epochs': 20,
                    'lr': 0.001
                }
            },
            # Lightweight Models
            {
                'name': 'Lightweight_Basic',
                'category': 'lightweight',
                'type': 'basic',
                'enabled': True,
                'params': {
                    'hidden_sizes': [128, 64],
                    'use_attention': True,
                    'dropout': 0.1
                },
                'training': {
                    'epochs': 15,
                    'lr': 0.001
                }
            },
            {
                'name': 'Lightweight_Conv',
                'category': 'lightweight',
                'type': 'conv',
                'enabled': True,
                'params': {
                    'use_attention': True,
                    'dropout': 0.1
                },
                'training': {
                    'epochs': 15,
                    'lr': 0.001
                }
            },
            {
                'name': 'Lightweight_Adaptive',
                'category': 'lightweight',
                'type': 'adaptive',
                'enabled': True,
                'params': {
                    'num_experts': 4,
                    'expert_size': 32
                },
                'training': {
                    'epochs': 20,
                    'lr': 0.001
                }
            },
            # Lightweight variants for ablation study
            {
                'name': 'Lightweight_No_Attention',
                'category': 'lightweight',
                'type': 'basic',
                'enabled': True,
                'params': {
                    'hidden_sizes': [128, 64],
                    'use_attention': False,
                    'dropout': 0.1
                },
                'training': {
                    'epochs': 15,
                    'lr': 0.001
                }
            }
        ]
    }


def run_quick_test():
    """Run a quick test with reduced epochs for development/testing."""
    
    config = get_default_config()
    
    # Reduce epochs for quick testing
    for model_config in config['models']:
        model_config['training']['epochs'] = 3
    
    # Enable only a subset of models for quick testing
    enabled_models = ['Traditional_CNN', 'Lightweight_Basic']
    for model_config in config['models']:
        model_config['enabled'] = model_config['name'] in enabled_models
    
    print("Running quick test (3 epochs, 2 models)...")
    return run_comparison_experiment(config)


def run_ablation_study():
    """Run ablation study focusing on lightweight models."""
    
    config = {
        'base_results_dir': './results',
        'data': {
            'batch_size': 64,
            'num_workers': 2
        },
        'models': [
            # Baseline traditional model
            {
                'name': 'Traditional_CNN_Baseline',
                'category': 'traditional',
                'type': 'cnn',
                'enabled': True,
                'params': {'dropout': 0.25},
                'training': {'epochs': 15, 'lr': 0.001}
            },
            # Lightweight variations
            {
                'name': 'Lightweight_With_Attention',
                'category': 'lightweight',
                'type': 'basic',
                'enabled': True,
                'params': {
                    'hidden_sizes': [128, 64],
                    'use_attention': True,
                    'attention_size': 64,
                    'dropout': 0.1
                },
                'training': {'epochs': 15, 'lr': 0.001}
            },
            {
                'name': 'Lightweight_Without_Attention',
                'category': 'lightweight',
                'type': 'basic',
                'enabled': True,
                'params': {
                    'hidden_sizes': [128, 64],
                    'use_attention': False,
                    'dropout': 0.1
                },
                'training': {'epochs': 15, 'lr': 0.001}
            },
            {
                'name': 'Lightweight_Small_Attention',
                'category': 'lightweight',
                'type': 'basic',
                'enabled': True,
                'params': {
                    'hidden_sizes': [128, 64],
                    'use_attention': True,
                    'attention_size': 32,
                    'dropout': 0.1
                },
                'training': {'epochs': 15, 'lr': 0.001}
            },
            {
                'name': 'Lightweight_Large_Network',
                'category': 'lightweight',
                'type': 'basic',
                'enabled': True,
                'params': {
                    'hidden_sizes': [256, 128, 64],
                    'use_attention': True,
                    'attention_size': 64,
                    'dropout': 0.1
                },
                'training': {'epochs': 15, 'lr': 0.001}
            }
        ]
    }
    
    print("Running ablation study on lightweight models...")
    return run_comparison_experiment(config)


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='MNIST Neural Network Architecture Comparison')
    parser.add_argument('--mode', choices=['full', 'quick', 'ablation'], default='full',
                       help='Experiment mode')
    parser.add_argument('--config', type=str, help='Path to custom config JSON file')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Base directory for results')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Override epochs for all models')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.results_dir:
        config['base_results_dir'] = args.results_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        for model_config in config['models']:
            model_config['training']['epochs'] = args.epochs
    
    # Run experiment based on mode
    if args.mode == 'quick':
        results_dir, results = run_quick_test()
    elif args.mode == 'ablation':
        results_dir, results = run_ablation_study()
    else:  # full
        results_dir, results = run_comparison_experiment(config)
    
    print(f"\nExperiment completed!")
    print(f"Results directory: {results_dir}")
    print(f"Models trained: {len(results)}")


if __name__ == "__main__":
    main()
