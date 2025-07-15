#!/bin/bash

# MNIST Neural Network Architecture Comparison Experiment
# Easy-to-use script for running experiments

set -e

echo "MNIST Neural Network Architecture Comparison"
echo "==========================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the mnist_experiment directory"
    exit 1
fi

# Function to install dependencies
install_deps() {
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    echo "Dependencies installed successfully!"
}

# Function to run setup test
test_setup() {
    echo "Testing experiment setup..."
    python test_setup.py
}

# Function to run quick experiment
run_quick() {
    echo "Running quick experiment (3 epochs, 2 models)..."
    python experiments/compare.py --mode quick
}

# Function to run full experiment
run_full() {
    echo "Running full experiment (all models, default epochs)..."
    python experiments/compare.py --mode full
}

# Function to run ablation study
run_ablation() {
    echo "Running ablation study (lightweight model variations)..."
    python experiments/compare.py --mode ablation
}

# Main menu
case "${1:-menu}" in
    "install")
        install_deps
        ;;
    "test")
        test_setup
        ;;
    "quick")
        run_quick
        ;;
    "full")
        run_full
        ;;
    "ablation")
        run_ablation
        ;;
    "menu"|*)
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  install   - Install required Python packages"
        echo "  test      - Test the experiment setup"
        echo "  quick     - Run quick experiment (3 epochs, 2 models)"
        echo "  full      - Run full experiment (all models)"
        echo "  ablation  - Run ablation study (lightweight variations)"
        echo ""
        echo "Examples:"
        echo "  $0 install   # First time setup"
        echo "  $0 test      # Verify everything works"
        echo "  $0 quick     # Quick test run"
        echo "  $0 full      # Full comparison"
        ;;
esac
