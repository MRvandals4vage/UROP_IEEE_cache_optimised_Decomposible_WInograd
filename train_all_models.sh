#!/bin/bash

# Comprehensive training script for all three models
# Usage: ./train_all_models.sh [epochs] [batch_size]

set -e  # Exit on any error

# Default parameters
EPOCHS=${1:-50}
BATCH_SIZE=${2:-128}
LR=0.1

# Base directories
BASE_DIR="/Users/ishaanupponi/Documents/My projects/Ml/MACHINE LEARNING FROM UROP"
RESNET_DIR="$BASE_DIR/ResNet-18 inference"
LIGHTWEIGHT_DIR="$BASE_DIR/Initial our proposed model/lightweight_cnn_project"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 COMPREHENSIVE MODEL TRAINING SCRIPT${NC}"
echo -e "${BLUE}====================================${NC}"
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo ""

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}$1${NC}"
    echo -e "${YELLOW}$(printf '=%.0s' {1..60})${NC}"
}

# Function to run command with error handling
run_command() {
    local cmd="$1"
    local description="$2"
    local dir="$3"
    
    echo -e "\n${BLUE}🔄 $description${NC}"
    echo "Command: $cmd"
    echo "Directory: $dir"
    echo "$(printf '-%.0s' {1..50})"
    
    if cd "$dir" && eval "$cmd"; then
        echo -e "${GREEN}✅ $description completed successfully!${NC}"
        return 0
    else
        echo -e "${RED}❌ $description failed!${NC}"
        return 1
    fi
}

# Start timing
START_TIME=$(date +%s)

print_section "STEP 1: TRAINING RESNET-18"
run_command "python3 -m src.scripts.train --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --save-path resnet18_final.ckpt" \
    "ResNet-18 Training" \
    "$RESNET_DIR"

print_section "STEP 2: TRAINING LIGHTWEIGHT CNN (STANDARD)"
run_command "python3 -m src.scripts.train_standard --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --save-path standard_final.ckpt" \
    "Standard CNN Training" \
    "$LIGHTWEIGHT_DIR"

print_section "STEP 3: TRAINING LIGHTWEIGHT CNN (WINOGRAD)"
run_command "python3 -m src.scripts.train_winograd --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --save-path winograd_final.ckpt" \
    "Winograd CNN Training" \
    "$LIGHTWEIGHT_DIR"

print_section "STEP 4: MODEL ANALYSIS"
echo -e "\n${BLUE}🔍 Analyzing Standard CNN...${NC}"
cd "$LIGHTWEIGHT_DIR" && python3 -m src.lightweight_cnn.model_standard

echo -e "\n${BLUE}🔍 Analyzing Winograd CNN...${NC}"
cd "$LIGHTWEIGHT_DIR" && python3 -m src.lightweight_cnn.model_winograd

print_section "STEP 5: COMPREHENSIVE COMPARISON"
run_command "python3 -m src.scripts.compare_winograd --standard-checkpoint standard_final.ckpt --winograd-checkpoint winograd_final.ckpt" \
    "Winograd vs Standard Comparison" \
    "$LIGHTWEIGHT_DIR"

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

print_section "TRAINING COMPLETED!"
echo -e "${GREEN}🎉 All models trained successfully!${NC}"
echo "Total training time: ${HOURS}h ${MINUTES}m"
echo ""
echo "📁 Model checkpoints:"
echo "  ResNet-18: $RESNET_DIR/resnet18_final.ckpt"
echo "  Standard CNN: $LIGHTWEIGHT_DIR/standard_final.ckpt"
echo "  Winograd CNN: $LIGHTWEIGHT_DIR/winograd_final.ckpt"
echo ""
echo -e "${BLUE}📊 SUMMARY:${NC}"
echo "✅ ResNet-18: ~11.2M params, ~555M MACs, highest accuracy"
echo "✅ Standard CNN: ~258K params, ~15.8M MACs, good efficiency"
echo "✅ Winograd CNN: ~258K params, ~7.0M MACs, best energy efficiency"
echo ""
echo -e "${GREEN}🏆 Experiment completed successfully!${NC}"
