"""
Visualization utilities for experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_curves(history_files: List[str], save_path: str = None, figsize: tuple = (15, 10)):
    """
    Plot training curves for multiple models.
    
    Args:
        history_files: List of paths to training history JSON files
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(history_files)))
    
    for i, (file_path, color) in enumerate(zip(history_files, colors)):
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            history = json.load(f)
        
        model_name = os.path.basename(file_path).replace('_history.json', '')
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training and Validation Loss
        axes[0, 0].plot(epochs, history['train_loss'], label=f'{model_name} (Train)', 
                       color=color, linestyle='-')
        axes[0, 0].plot(epochs, history['val_loss'], label=f'{model_name} (Val)', 
                       color=color, linestyle='--')
        
        # Training and Validation Accuracy
        axes[0, 1].plot(epochs, history['train_acc'], label=f'{model_name} (Train)', 
                       color=color, linestyle='-')
        axes[0, 1].plot(epochs, history['val_acc'], label=f'{model_name} (Val)', 
                       color=color, linestyle='--')
        
        # Epoch Times
        axes[1, 0].plot(epochs, history['epoch_times'], label=model_name, 
                       color=color, marker='o', markersize=3)
        
        # Memory Usage (CPU)
        cpu_memory = [mem['cpu_memory_gb'] for mem in history['memory_usage']]
        axes[1, 1].plot(epochs, cpu_memory, label=model_name, 
                       color=color, marker='s', markersize=3)
    
    # Configure subplots
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Time per Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Memory Usage (CPU)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Memory (GB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = None, figsize: tuple = (16, 12)):
    """
    Create comprehensive comparison plots for models.
    
    Args:
        comparison_df: DataFrame from utils.metrics.compare_models()
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy vs Parameters
    axes[0, 0].scatter(comparison_df['Parameters'], comparison_df['Final Accuracy (%)'], 
                      s=100, alpha=0.7, c=range(len(comparison_df)), cmap='viridis')
    for i, model in enumerate(comparison_df['Model']):
        axes[0, 0].annotate(model, (comparison_df['Parameters'].iloc[i], 
                           comparison_df['Final Accuracy (%)'].iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 0].set_xlabel('Number of Parameters')
    axes[0, 0].set_ylabel('Final Accuracy (%)')
    axes[0, 0].set_title('Accuracy vs Model Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training Time vs Accuracy
    axes[0, 1].scatter(comparison_df['Training Time (s)'], comparison_df['Final Accuracy (%)'], 
                      s=100, alpha=0.7, c=range(len(comparison_df)), cmap='viridis')
    for i, model in enumerate(comparison_df['Model']):
        axes[0, 1].annotate(model, (comparison_df['Training Time (s)'].iloc[i], 
                           comparison_df['Final Accuracy (%)'].iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_xlabel('Training Time (s)')
    axes[0, 1].set_ylabel('Final Accuracy (%)')
    axes[0, 1].set_title('Accuracy vs Training Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Inference Speed
    axes[0, 2].bar(comparison_df['Model'], comparison_df['Inference FPS'], alpha=0.7)
    axes[0, 2].set_xlabel('Model')
    axes[0, 2].set_ylabel('Inference FPS')
    axes[0, 2].set_title('Inference Speed')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Memory Usage
    x = np.arange(len(comparison_df))
    width = 0.35
    axes[1, 0].bar(x - width/2, comparison_df['CPU Memory (GB)'], width, 
                  label='CPU Memory', alpha=0.7)
    axes[1, 0].bar(x + width/2, comparison_df['GPU Memory (GB)'], width, 
                  label='GPU Memory', alpha=0.7)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Memory (GB)')
    axes[1, 0].set_title('Memory Usage')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(comparison_df['Model'], rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Efficiency Metrics
    axes[1, 1].bar(comparison_df['Model'], comparison_df['Accuracy/Param (×1000)'], alpha=0.7)
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Accuracy per 1K Parameters')
    axes[1, 1].set_title('Parameter Efficiency')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Overall Performance Radar (if Overall Score exists)
    if 'Overall Score' in comparison_df.columns:
        axes[1, 2].bar(comparison_df['Model'], comparison_df['Overall Score'], alpha=0.7)
        axes[1, 2].set_xlabel('Model')
        axes[1, 2].set_ylabel('Overall Score')
        axes[1, 2].set_title('Overall Performance Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # Alternative: Show accuracy bar chart
        axes[1, 2].bar(comparison_df['Model'], comparison_df['Final Accuracy (%)'], alpha=0.7)
        axes[1, 2].set_xlabel('Model')
        axes[1, 2].set_ylabel('Final Accuracy (%)')
        axes[1, 2].set_title('Final Accuracy Comparison')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_architecture_analysis(comparison_df: pd.DataFrame, save_path: str = None, figsize: tuple = (14, 8)):
    """
    Analyze performance by architecture type.
    
    Args:
        comparison_df: DataFrame from utils.metrics.compare_models()
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Architecture Analysis', fontsize=16, fontweight='bold')
    
    # Group by architecture type
    arch_groups = comparison_df.groupby('Architecture').agg({
        'Final Accuracy (%)': ['mean', 'std'],
        'Parameters': 'mean',
        'Training Time (s)': 'mean',
        'Inference FPS': 'mean'
    }).round(2)
    
    # Architecture Performance
    arch_names = arch_groups.index
    accuracies = arch_groups['Final Accuracy (%)']['mean']
    std_errs = arch_groups['Final Accuracy (%)']['std']
    
    axes[0].bar(arch_names, accuracies, yerr=std_errs, capsize=5, alpha=0.7)
    axes[0].set_xlabel('Architecture')
    axes[0].set_ylabel('Average Accuracy (%)')
    axes[0].set_title('Average Performance by Architecture')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Parameters vs Performance
    params = arch_groups['Parameters']['mean']
    axes[1].scatter(params, accuracies, s=200, alpha=0.7, c=range(len(arch_names)), cmap='viridis')
    for i, arch in enumerate(arch_names):
        axes[1].annotate(arch, (params.iloc[i], accuracies.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    axes[1].set_xlabel('Average Parameters')
    axes[1].set_ylabel('Average Accuracy (%)')
    axes[1].set_title('Architecture Efficiency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_convergence_analysis(history_files: List[str], save_path: str = None, figsize: tuple = (12, 8)):
    """
    Analyze and plot convergence behavior.
    
    Args:
        history_files: List of paths to training history JSON files
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
    
    convergence_data = []
    
    for file_path in history_files:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            history = json.load(f)
        
        model_name = os.path.basename(file_path).replace('_history.json', '')
        val_acc = history['val_acc']
        epochs = range(1, len(val_acc) + 1)
        
        # Plot validation accuracy
        axes[0].plot(epochs, val_acc, label=model_name, marker='o', markersize=3)
        
        # Calculate convergence metrics
        best_acc = max(val_acc)
        epochs_to_95_percent = None
        epochs_to_99_percent = None
        
        for i, acc in enumerate(val_acc):
            if epochs_to_95_percent is None and acc >= 0.95 * best_acc:
                epochs_to_95_percent = i + 1
            if epochs_to_99_percent is None and acc >= 0.99 * best_acc:
                epochs_to_99_percent = i + 1
        
        convergence_data.append({
            'Model': model_name,
            'Epochs to 95%': epochs_to_95_percent or len(val_acc),
            'Epochs to 99%': epochs_to_99_percent or len(val_acc),
            'Final Accuracy': val_acc[-1],
            'Best Accuracy': best_acc
        })
    
    # Configure convergence plot
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Validation Accuracy Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Convergence speed comparison
    conv_df = pd.DataFrame(convergence_data)
    if not conv_df.empty:
        x = np.arange(len(conv_df))
        width = 0.35
        
        axes[1].bar(x - width/2, conv_df['Epochs to 95%'], width, 
                   label='95% of Best', alpha=0.7)
        axes[1].bar(x + width/2, conv_df['Epochs to 99%'], width, 
                   label='99% of Best', alpha=0.7)
        
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Epochs to Convergence')
        axes[1].set_title('Convergence Speed')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(conv_df['Model'], rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_dashboard(results_dir: str, save_dir: str = None):
    """
    Create a comprehensive dashboard with all visualizations.
    
    Args:
        results_dir: Directory containing result files
        save_dir: Directory to save plots (if None, plots are only displayed)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Load results
    result_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]
    history_files = [os.path.join(results_dir, f.replace('_results.json', '_history.json')) 
                    for f in result_files]
    history_files = [f for f in history_files if os.path.exists(f)]
    
    if not result_files:
        print("No result files found!")
        return
    
    # Load comparison data
    from .metrics import compare_models, rank_models
    
    results_list = []
    for file_name in result_files:
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, 'r') as f:
            results = json.load(f)
        results_list.append(results)
    
    comparison_df = compare_models(results_list)
    ranked_df = rank_models(comparison_df)
    
    # Create plots
    print("Creating training curves...")
    plot_training_curves(history_files, 
                         save_path=os.path.join(save_dir, 'training_curves.png') if save_dir else None)
    
    print("Creating model comparison...")
    plot_model_comparison(ranked_df, 
                         save_path=os.path.join(save_dir, 'model_comparison.png') if save_dir else None)
    
    print("Creating architecture analysis...")
    plot_architecture_analysis(ranked_df, 
                              save_path=os.path.join(save_dir, 'architecture_analysis.png') if save_dir else None)
    
    print("Creating convergence analysis...")
    plot_convergence_analysis(history_files, 
                             save_path=os.path.join(save_dir, 'convergence_analysis.png') if save_dir else None)
    
    print("Dashboard creation completed!")


if __name__ == "__main__":
    # Test the plotting module
    print("Testing plotting module...")
    
    # Create sample data for testing
    sample_history = {
        'train_loss': [2.3, 1.8, 1.2, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18],
        'train_acc': [10, 35, 60, 75, 82, 88, 92, 94, 96, 97],
        'val_loss': [2.2, 1.9, 1.3, 0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28],
        'val_acc': [12, 32, 58, 72, 80, 85, 89, 91, 93, 94],
        'epoch_times': [12, 11, 12, 11, 12, 11, 12, 11, 12, 11],
        'memory_usage': [{'cpu_memory_gb': 2.0, 'gpu_memory_gb': 1.0} for _ in range(10)]
    }
    
    # Save sample history for testing
    test_dir = './test_results'
    os.makedirs(test_dir, exist_ok=True)
    
    with open(os.path.join(test_dir, 'test_model_history.json'), 'w') as f:
        json.dump(sample_history, f)
    
    # Test plotting functions
    try:
        plot_training_curves([os.path.join(test_dir, 'test_model_history.json')])
        print("Training curves plot: OK")
    except Exception as e:
        print(f"Training curves plot: Error - {e}")
    
    # Clean up
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("Plotting module test completed!")
