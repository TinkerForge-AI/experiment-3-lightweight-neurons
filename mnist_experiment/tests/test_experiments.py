import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments import compare

def test_get_default_config():
    config = compare.get_default_config()
    assert 'models' in config
    assert 'data' in config
    assert isinstance(config['models'], list)
    assert isinstance(config['data'], dict)
