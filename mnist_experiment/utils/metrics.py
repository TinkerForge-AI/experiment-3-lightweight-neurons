"""
Performance metrics and analysis utilities.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Union
import os


def calculate_efficiency_metrics(results: Dict) -> Dict:
    """
    Calculate efficiency metrics from training results.
    
    Args:
        results: Dictionary containing training results
        
    Returns:
        Dict: Efficiency metrics
    """
    import logging
    model_info = results.get('model_info', {})
    total_params = model_info.get('total_params', 0)
    try:
        efficiency_metrics = {
            'accuracy_per_param': results['final_accuracy'] / max(total_params, 1) * 1000,  # Accuracy per 1K params
            'accuracy_per_second': results['final_accuracy'] / max(results['total_training_time'], 1),  # Accuracy per training second
            'params_efficiency': total_params / max(results['final_accuracy'], 1),  # Params needed per 1% accuracy
            'time_efficiency': results['total_training_time'] / max(results['final_accuracy'], 1),  # Time needed per 1% accuracy
            'inference_fps': 1.0 / max(results['avg_batch_inference_time'], 1e-6),  # Frames per second
            'memory_efficiency': results['final_accuracy'] / max(results['final_memory_usage']['cpu_memory_gb'], 0.1)  # Accuracy per GB
        }
    except Exception as e:
        logging.error(f"Error in calculate_efficiency_metrics: {e}")
        raise
    return efficiency_metrics


def compare_models(results_list: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple models and create a comparison dataframe.
    
    Args:
        results_list: List of result dictionaries from different models
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for results in results_list:
        model_info = results.get('model_info', {})
        efficiency = calculate_efficiency_metrics(results)
        
        row = {
            'Model': results['model_name'],
            'Architecture': model_info.get('architecture', 'Unknown'),
            'Parameters': model_info.get('total_params', 0),
            'Final Accuracy (%)': results['final_accuracy'],
            'Best Accuracy (%)': results['best_accuracy'],
            'Training Time (s)': results['total_training_time'],
            'Avg Epoch Time (s)': results['avg_epoch_time'],
            'Inference Time (s)': results['avg_batch_inference_time'],
            'Inference FPS': efficiency['inference_fps'],
            'CPU Memory (GB)': results['final_memory_usage']['cpu_memory_gb'],
            'GPU Memory (GB)': results['final_memory_usage']['gpu_memory_gb'],
            'Accuracy/Param (×1000)': efficiency['accuracy_per_param'],
            'Accuracy/Time': efficiency['accuracy_per_second'],
            'Memory Efficiency': efficiency['memory_efficiency']
        }
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def rank_models(comparison_df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Rank models based on weighted criteria.
    
    Args:
        comparison_df: DataFrame from compare_models()
        weights: Dictionary of weights for each metric
        
    Returns:
        pd.DataFrame: Ranked comparison with scores
    """
    if weights is None:
        weights = {
            'Final Accuracy (%)': 0.4,
            'Training Time (s)': -0.2,  # Negative because lower is better
            'Inference FPS': 0.2,
            'Parameters': -0.1,  # Negative because fewer is better
            'Memory Efficiency': 0.1
        }
    
    # Normalize metrics to 0-1 scale
    normalized_df = comparison_df.copy()
    
    for metric, weight in weights.items():
        if metric in normalized_df.columns:
            if weight > 0:  # Higher is better
                normalized_df[f'{metric}_norm'] = (
                    (normalized_df[metric] - normalized_df[metric].min()) /
                    (normalized_df[metric].max() - normalized_df[metric].min() + 1e-8)
                )
            else:  # Lower is better
                normalized_df[f'{metric}_norm'] = (
                    (normalized_df[metric].max() - normalized_df[metric]) /
                    (normalized_df[metric].max() - normalized_df[metric].min() + 1e-8)
                )
    
    # Calculate weighted score
    scores = []
    for idx, row in normalized_df.iterrows():
        score = 0
        for metric, weight in weights.items():
            if f'{metric}_norm' in normalized_df.columns:
                score += abs(weight) * row[f'{metric}_norm']
        scores.append(score)
    
    comparison_df['Overall Score'] = scores
    comparison_df['Rank'] = comparison_df['Overall Score'].rank(ascending=False)
    
    # Sort by rank
    ranked_df = comparison_df.sort_values('Rank')
    
    return ranked_df


def analyze_training_convergence(history_files: List[str]) -> Dict:
    """
    Analyze training convergence from history files.
    
    Args:
        history_files: List of paths to training history JSON files
        
    Returns:
        Dict: Convergence analysis
    """
    convergence_analysis = {}
    
    for file_path in history_files:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            history = json.load(f)
        
        model_name = os.path.basename(file_path).replace('_history.json', '')
        
        train_acc = history['train_acc']
        val_acc = history['val_acc']
        epoch_times = history['epoch_times']
        
        # Find convergence point (where validation accuracy stops improving significantly)
        convergence_epoch = len(val_acc)  # Default to last epoch
        best_val_acc = max(val_acc)
        
        for i in range(len(val_acc) - 1, -1, -1):
            if val_acc[i] >= best_val_acc - 0.5:  # Within 0.5% of best
                convergence_epoch = i + 1
                break
        
        # Calculate metrics
        analysis = {
            'convergence_epoch': convergence_epoch,
            'epochs_to_90_percent_best': None,
            'final_train_acc': train_acc[-1],
            'final_val_acc': val_acc[-1],
            'best_val_acc': best_val_acc,
            'overfitting_score': max(0, train_acc[-1] - val_acc[-1]),
            'avg_epoch_time': np.mean(epoch_times),
            'training_stability': np.std(val_acc[-5:]) if len(val_acc) >= 5 else np.std(val_acc)
        }
        
        # Find when model reached 90% of best accuracy
        target_acc = 0.9 * best_val_acc
        for i, acc in enumerate(val_acc):
            if acc >= target_acc:
                analysis['epochs_to_90_percent_best'] = i + 1
                break
        
        convergence_analysis[model_name] = analysis
    
    return convergence_analysis


def calculate_statistical_significance(results1: Dict, results2: Dict, metric: str = 'final_accuracy') -> Dict:
    """
    Calculate statistical significance between two models (placeholder for multiple runs).
    
    Args:
        results1: Results from model 1
        results2: Results from model 2
        metric: Metric to compare
        
    Returns:
        Dict: Statistical comparison
    """
    # This is a simplified version - in practice, you'd run multiple seeds
    # and use proper statistical tests
    
    value1 = results1.get(metric, 0)
    value2 = results2.get(metric, 0)
    
    difference = abs(value1 - value2)
    relative_difference = difference / max(value1, value2, 1e-8) * 100
    
    # Simple threshold-based significance (replace with proper tests in real use)
    is_significant = relative_difference > 1.0  # 1% threshold
    
    return {
        'model1_value': value1,
        'model2_value': value2,
        'absolute_difference': difference,
        'relative_difference_percent': relative_difference,
        'is_significant': is_significant,
        'winner': 'Model 1' if value1 > value2 else 'Model 2'
    }


def generate_summary_report(results_dir: str, output_file: str = None) -> str:
    """
    Generate a comprehensive summary report from all results in a directory.
    
    Args:
        results_dir: Directory containing result files
        output_file: Optional file to save the report
        
    Returns:
        str: Summary report text
    """
    # Load all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]
    results_list = []
    
    for file_name in result_files:
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, 'r') as f:
            results = json.load(f)
        results_list.append(results)
    
    if not results_list:
        return "No results found in the specified directory."
    
    # Generate comparisons
    comparison_df = compare_models(results_list)
    ranked_df = rank_models(comparison_df)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("MNIST NEURAL NETWORK ARCHITECTURE COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY STATISTICS")
    report.append("-" * 40)
    report.append(f"Number of models compared: {len(results_list)}")
    report.append(f"Best accuracy achieved: {comparison_df['Best Accuracy (%)'].max():.2f}%")
    report.append(f"Average accuracy: {comparison_df['Final Accuracy (%)'].mean():.2f}%")
    report.append(f"Fastest training time: {comparison_df['Training Time (s)'].min():.2f}s")
    report.append(f"Fastest inference: {comparison_df['Inference FPS'].max():.2f} FPS")
    report.append("")
    
    # Top performers
    report.append("TOP PERFORMERS")
    report.append("-" * 40)
    top_3 = ranked_df.head(3)
    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
        report.append(f"{idx}. {row['Model']} (Score: {row['Overall Score']:.3f})")
        report.append(f"   Accuracy: {row['Final Accuracy (%)']:.2f}%")
        report.append(f"   Parameters: {row['Parameters']:,}")
        report.append(f"   Training Time: {row['Training Time (s)']:.2f}s")
        report.append("")
    
    # Detailed comparison table
    report.append("DETAILED COMPARISON")
    report.append("-" * 40)
    report.append(ranked_df.to_string(index=False))
    report.append("")
    
    # Efficiency analysis
    report.append("EFFICIENCY ANALYSIS")
    report.append("-" * 40)
    most_efficient_param = ranked_df.loc[ranked_df['Accuracy/Param (×1000)'].idxmax()]
    fastest_training = ranked_df.loc[ranked_df['Training Time (s)'].idxmin()]
    best_memory_eff = ranked_df.loc[ranked_df['Memory Efficiency'].idxmax()]
    
    report.append(f"Most parameter-efficient: {most_efficient_param['Model']}")
    report.append(f"Fastest training: {fastest_training['Model']}")
    report.append(f"Best memory efficiency: {best_memory_eff['Model']}")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    best_overall = ranked_df.iloc[0]
    report.append(f"Best overall model: {best_overall['Model']}")
    report.append(f"Recommended for production use based on balanced performance.")
    report.append("")
    
    if len(ranked_df) > 1:
        traditional_models = ranked_df[ranked_df['Architecture'].str.contains('Traditional|Convolutional|Residual', na=False)]
        lightweight_models = ranked_df[ranked_df['Architecture'].str.contains('Lightweight|Single-Weight|Attention', na=False)]
        
        if not traditional_models.empty and not lightweight_models.empty:
            best_traditional = traditional_models.iloc[0]
            best_lightweight = lightweight_models.iloc[0]
            
            report.append("ARCHITECTURE COMPARISON")
            report.append("-" * 40)
            report.append(f"Best Traditional: {best_traditional['Model']} (Acc: {best_traditional['Final Accuracy (%)']:.2f}%)")
            report.append(f"Best Lightweight: {best_lightweight['Model']} (Acc: {best_lightweight['Final Accuracy (%)']:.2f}%)")
            
            param_ratio = best_traditional['Parameters'] / best_lightweight['Parameters']
            time_ratio = best_traditional['Training Time (s)'] / best_lightweight['Training Time (s)']
            
            report.append(f"Parameter ratio (Traditional/Lightweight): {param_ratio:.2f}x")
            report.append(f"Training time ratio (Traditional/Lightweight): {time_ratio:.2f}x")
    
    report_text = "\n".join(report)
    
    # Save report if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
    
    return report_text


if __name__ == "__main__":
    # Test the metrics module
    print("Testing metrics module...")
    
    # Create sample results for testing
    sample_results = [
        {
            'model_name': 'Traditional CNN',
            'final_accuracy': 98.5,
            'best_accuracy': 98.8,
            'total_training_time': 120.0,
            'avg_epoch_time': 12.0,
            'avg_batch_inference_time': 0.01,
            'final_memory_usage': {'cpu_memory_gb': 2.0, 'gpu_memory_gb': 1.0},
            'model_info': {'total_params': 50000, 'architecture': 'Traditional Convolutional'}
        },
        {
            'model_name': 'Lightweight Network',
            'final_accuracy': 97.8,
            'best_accuracy': 98.2,
            'total_training_time': 80.0,
            'avg_epoch_time': 8.0,
            'avg_batch_inference_time': 0.008,
            'final_memory_usage': {'cpu_memory_gb': 1.5, 'gpu_memory_gb': 0.5},
            'model_info': {'total_params': 15000, 'architecture': 'Lightweight Single-Weight Neurons'}
        }
    ]
    
    # Test comparison
    comparison_df = compare_models(sample_results)
    print("Model Comparison:")
    print(comparison_df)
    print()
    
    # Test ranking
    ranked_df = rank_models(comparison_df)
    print("Ranked Models:")
    print(ranked_df[['Model', 'Final Accuracy (%)', 'Overall Score', 'Rank']])
    print()
    
    # Test efficiency metrics
    for results in sample_results:
        efficiency = calculate_efficiency_metrics(results)
        print(f"Efficiency metrics for {results['model_name']}:")
        for metric, value in efficiency.items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    print("Metrics module test completed!")
