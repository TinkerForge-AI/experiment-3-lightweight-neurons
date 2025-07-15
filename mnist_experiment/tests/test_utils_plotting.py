import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plotting
import tempfile
import json

def test_plot_training_curves_runs():
    # Create a fake history file
    with tempfile.TemporaryDirectory() as tmpdir:
        history = {
            'train_loss': [2.0, 1.0],
            'train_acc': [10, 90],
            'val_loss': [2.1, 1.1],
            'val_acc': [12, 88],
            'epoch_times': [1, 1],
            'memory_usage': [{'cpu_memory_gb': 1.0, 'gpu_memory_gb': 0.5}, {'cpu_memory_gb': 1.1, 'gpu_memory_gb': 0.6}]
        }
        path = os.path.join(tmpdir, 'test_history.json')
        with open(path, 'w') as f:
            json.dump(history, f)
        # Should not raise
        plotting.plot_training_curves([path])
