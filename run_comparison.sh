#!/bin/bash

# Complete comparison script for Lightweight CNN vs ResNet-18
# This script helps you run all experiments and comparisons

set -e  # Exit on error

echo "=========================================="
echo "Model Comparison: Lightweight CNN vs ResNet-18"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIGHTWEIGHT_DIR="$SCRIPT_DIR/Initial our proposed model/lightweight_cnn_project"
RESNET_DIR="$SCRIPT_DIR/ResNet-18 inference"

# Parse command line arguments
ACTION=${1:-help}

case $ACTION in
    "stats")
        print_step "Measuring model statistics (no training required)..."
        echo ""
        
        print_info "Lightweight CNN Statistics:"
        cd "$LIGHTWEIGHT_DIR"
        python3 -m src.scripts.measure_params_macs
        
        echo ""
        print_info "ResNet-18 Statistics:"
        cd "$RESNET_DIR"
        python3 -m src.scripts.measure_params_macs
        
        print_success "Statistics measurement complete!"
        ;;
    
    "visualize")
        print_step "Visualizing architecture differences..."
        cd "$RESNET_DIR"
        python3 -m src.scripts.visualize_architecture
        print_success "Visualization complete!"
        ;;
    
    "train-light")
        EPOCHS=${2:-100}
        BATCH_SIZE=${3:-128}
        
        print_step "Training Lightweight CNN for $EPOCHS epochs..."
        cd "$LIGHTWEIGHT_DIR"
        python3 -m src.scripts.train --epochs $EPOCHS --batch-size $BATCH_SIZE
        print_success "Lightweight CNN training complete!"
        ;;
    
    "train-resnet")
        EPOCHS=${2:-100}
        BATCH_SIZE=${3:-128}
        
        print_step "Training ResNet-18 for $EPOCHS epochs..."
        cd "$RESNET_DIR"
        python3 -m src.scripts.train --epochs $EPOCHS --batch-size $BATCH_SIZE
        print_success "ResNet-18 training complete!"
        ;;
    
    "train-both")
        EPOCHS=${2:-100}
        BATCH_SIZE=${3:-128}
        
        print_step "Training both models for $EPOCHS epochs..."
        
        print_info "Step 1/2: Training Lightweight CNN..."
        cd "$LIGHTWEIGHT_DIR"
        python3 -m src.scripts.train --epochs $EPOCHS --batch-size $BATCH_SIZE
        
        print_info "Step 2/2: Training ResNet-18..."
        cd "$RESNET_DIR"
        python3 -m src.scripts.train --epochs $EPOCHS --batch-size $BATCH_SIZE
        
        print_success "Both models trained successfully!"
        ;;
    
    "eval-light")
        print_step "Evaluating Lightweight CNN..."
        cd "$LIGHTWEIGHT_DIR"
        python3 -m src.scripts.evaluate --checkpoint best_model.ckpt
        print_success "Evaluation complete!"
        ;;
    
    "eval-resnet")
        print_step "Evaluating ResNet-18..."
        cd "$RESNET_DIR"
        python3 -m src.scripts.evaluate --checkpoint best_resnet18.ckpt
        print_success "Evaluation complete!"
        ;;
    
    "compare")
        print_step "Comparing both models..."
        cd "$RESNET_DIR"
        python3 -m src.scripts.compare_models \
            --resnet-checkpoint best_resnet18.ckpt \
            --lightweight-checkpoint "$LIGHTWEIGHT_DIR/best_model.ckpt"
        print_success "Comparison complete!"
        ;;
    
    "full-experiment")
        EPOCHS=${2:-100}
        
        echo "=========================================="
        echo "Running Full Experiment Pipeline"
        echo "Epochs: $EPOCHS"
        echo "=========================================="
        echo ""
        
        print_step "1/5: Measuring initial statistics..."
        bash "$0" stats
        
        echo ""
        print_step "2/5: Visualizing architectures..."
        bash "$0" visualize
        
        echo ""
        print_step "3/5: Training Lightweight CNN..."
        bash "$0" train-light $EPOCHS
        
        echo ""
        print_step "4/5: Training ResNet-18..."
        bash "$0" train-resnet $EPOCHS
        
        echo ""
        print_step "5/5: Comparing models..."
        bash "$0" compare
        
        echo ""
        echo "=========================================="
        print_success "Full experiment pipeline complete!"
        echo "=========================================="
        ;;
    
    "quick-test")
        print_info "Running quick test (10 epochs)..."
        bash "$0" train-both 10
        bash "$0" compare
        ;;
    
    "help"|*)
        echo "Usage: bash run_comparison.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  stats              - Show model statistics (params, MACs) without training"
        echo "  visualize          - Visualize architecture differences"
        echo "  train-light [E] [B] - Train Lightweight CNN (E=epochs, B=batch_size)"
        echo "  train-resnet [E] [B] - Train ResNet-18 (E=epochs, B=batch_size)"
        echo "  train-both [E] [B]  - Train both models (E=epochs, B=batch_size)"
        echo "  eval-light         - Evaluate Lightweight CNN"
        echo "  eval-resnet        - Evaluate ResNet-18"
        echo "  compare            - Compare both trained models"
        echo "  full-experiment [E] - Run complete pipeline (stats → train → compare)"
        echo "  quick-test         - Quick test with 10 epochs"
        echo "  help               - Show this help message"
        echo ""
        echo "Examples:"
        echo "  bash run_comparison.sh stats"
        echo "  bash run_comparison.sh train-light 100 128"
        echo "  bash run_comparison.sh train-both 50"
        echo "  bash run_comparison.sh compare"
        echo "  bash run_comparison.sh full-experiment 100"
        echo "  bash run_comparison.sh quick-test"
        echo ""
        echo "Default values:"
        echo "  Epochs: 100"
        echo "  Batch size: 128"
        ;;
esac
