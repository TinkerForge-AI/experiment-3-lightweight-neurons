import pytest
import os
import sys
import tempfile
import json
import pandas as pd
# Ensure mnist_experiment is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import metrics

def test_calculate_efficiency_metrics_basic():
    results = {
        'final_accuracy': 95.0,
        'total_training_time': 100.0,
        'avg_batch_inference_time': 0.01,
        'final_memory_usage': {'cpu_memory_gb': 1.0, 'gpu_memory_gb': 0.5},
        'model_info': {'total_params': 10000}
    }
    eff = metrics.calculate_efficiency_metrics(results)
    assert eff['accuracy_per_param'] > 0
    assert eff['inference_fps'] > 0
    assert eff['memory_efficiency'] > 0

def test_compare_models_and_rank():
    results_list = [
        {
            'model_name': 'A',
            'final_accuracy': 90.0,
            'best_accuracy': 91.0,
            'total_training_time': 100.0,
            'avg_epoch_time': 10.0,
            'avg_batch_inference_time': 0.01,
            'final_memory_usage': {'cpu_memory_gb': 1.0, 'gpu_memory_gb': 0.5},
            'model_info': {'total_params': 10000, 'architecture': 'Test'}
        },
        {
            'model_name': 'B',
            'final_accuracy': 92.0,
            'best_accuracy': 93.0,
            'total_training_time': 80.0,
            'avg_epoch_time': 8.0,
            'avg_batch_inference_time': 0.008,
            'final_memory_usage': {'cpu_memory_gb': 1.5, 'gpu_memory_gb': 0.5},
            'model_info': {'total_params': 15000, 'architecture': 'Test'}
        }
    ]
    df = metrics.compare_models(results_list)
    assert isinstance(df, pd.DataFrame)
    ranked = metrics.rank_models(df)
    assert 'Overall Score' in ranked.columns
    assert 'Rank' in ranked.columns

def test_analyze_training_convergence():
    # Create a temporary history file
    with tempfile.TemporaryDirectory() as tmpdir:
        history = {
            'train_acc': [10, 50, 80, 90, 95],
            'val_acc': [12, 48, 78, 89, 94],
            'epoch_times': [10, 10, 10, 10, 10],
            'memory_usage': [{'cpu_memory_gb': 1.0, 'gpu_memory_gb': 0.5} for _ in range(5)]
        }
        path = os.path.join(tmpdir, 'model_history.json')
        with open(path, 'w') as f:
            json.dump(history, f)
        result = metrics.analyze_training_convergence([path])
        # The key is the filename without _history.json
        key = os.path.basename(path).replace('_history.json', '')
        assert key in result
        assert result[key]['convergence_epoch'] > 0

def test_statistical_significance():
    r1 = {'final_accuracy': 95.0}
    r2 = {'final_accuracy': 90.0}
    out = metrics.calculate_statistical_significance(r1, r2)
    assert out['is_significant'] is True
    assert out['winner'] == 'Model 1'

def test_generate_summary_report():
    # Create temporary result files
    with tempfile.TemporaryDirectory() as tmpdir:
        results = {
            'model_name': 'A',
            'final_accuracy': 90.0,
            'best_accuracy': 91.0,
            'total_training_time': 100.0,
            'avg_epoch_time': 10.0,
            'avg_batch_inference_time': 0.01,
            'final_memory_usage': {'cpu_memory_gb': 1.0, 'gpu_memory_gb': 0.5},
            'model_info': {'total_params': 10000, 'architecture': 'Test'}
        }
        path = os.path.join(tmpdir, 'A_results.json')
        with open(path, 'w') as f:
            json.dump(results, f)
        report = metrics.generate_summary_report(tmpdir)
        assert 'MNIST NEURAL NETWORK ARCHITECTURE COMPARISON REPORT' in report
