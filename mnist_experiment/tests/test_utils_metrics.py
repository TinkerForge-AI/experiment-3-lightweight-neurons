import pytest
import sys
import os
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
