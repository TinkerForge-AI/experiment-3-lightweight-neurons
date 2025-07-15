import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import traditional, lightweight

def test_traditional_models_forward():
    x = torch.randn(4, 1, 28, 28)
    for model_type in ['mlp', 'cnn', 'resnet']:
        model = traditional.get_traditional_model(model_type)
        out = model(x)
        assert out.shape[0] == 4
        assert out.shape[1] == 10

def test_lightweight_models_forward():
    x = torch.randn(4, 1, 28, 28)
    for model_type in ['basic', 'conv', 'adaptive']:
        model = lightweight.get_lightweight_model(model_type)
        out = model(x)
        assert out.shape[0] == 4
        assert out.shape[1] == 10
