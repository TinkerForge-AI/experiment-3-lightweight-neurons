import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import mnist_loader

def test_get_mnist_dataloaders():
    train_loader, test_loader = mnist_loader.get_mnist_dataloaders(batch_size=8)
    assert train_loader is not None
    assert test_loader is not None
    batch = next(iter(train_loader))
    images, labels = batch
    assert images.shape[0] == 8
    assert labels.shape[0] == 8

def test_get_mnist_info():
    info = mnist_loader.get_mnist_info()
    assert 'input_shape' in info
    assert info['num_classes'] == 10
