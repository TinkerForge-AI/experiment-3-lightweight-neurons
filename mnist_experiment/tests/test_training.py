import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from training import train
from models.traditional import TraditionalCNN
from data.mnist_loader import get_mnist_dataloaders

def test_model_trainer_memory_and_inference():
    train_loader, test_loader = get_mnist_dataloaders(batch_size=4)
    model = TraditionalCNN()
    trainer = train.ModelTrainer(model)
    mem = trainer.get_memory_usage()
    assert 'cpu_memory_gb' in mem
    avg_time, std_time = trainer.measure_inference_speed(test_loader, num_batches=2)
    assert avg_time > 0
    assert std_time >= 0
