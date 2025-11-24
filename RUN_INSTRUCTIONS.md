# 🚀 Complete Model Comparison Instructions

This guide shows you how to run all three models: **ResNet-18**, **Lightweight CNN (Standard)**, and **Lightweight CNN (Winograd)**.

## 📋 Available Scripts

### 1. **Full Training Script** (Recommended)
```bash
# Train all models with default settings (50 epochs)
./train_all_models.sh

# Train with custom epochs and batch size
./train_all_models.sh 100 128
```

### 2. **Python Training Script** (More Control)
```bash
# Train all models
python3 run_all_models.py --epochs 50 --batch-size 128

# Train only specific models
python3 run_all_models.py --skip-resnet --epochs 30
python3 run_all_models.py --skip-standard --skip-winograd

# Compare existing models only
python3 run_all_models.py --compare-only
```

### 3. **Quick Test** (5 minutes)
```bash
# Quick test without full training
python3 quick_comparison.py
```

## 🎯 Individual Model Training

### ResNet-18
```bash
cd "ResNet-18 inference"
python3 -m src.scripts.train --epochs 100 --batch-size 128 --save-path resnet18_model.ckpt
```

### Lightweight CNN (Standard)
```bash
cd "Initial our proposed model/lightweight_cnn_project"
python3 -m src.scripts.train_standard --epochs 100 --batch-size 128 --save-path standard_model.ckpt
```

### Lightweight CNN (Winograd)
```bash
cd "Initial our proposed model/lightweight_cnn_project"
python3 -m src.scripts.train_winograd --epochs 100 --batch-size 128 --save-path winograd_model.ckpt
```

## 📊 Model Analysis

### Test Model Architectures
```bash
cd "Initial our proposed model/lightweight_cnn_project"

# Test standard model
python3 -m src.lightweight_cnn.model_standard

# Test Winograd model  
python3 -m src.lightweight_cnn.model_winograd
```

### Compare Winograd vs Standard
```bash
cd "Initial our proposed model/lightweight_cnn_project"
python3 -m src.scripts.compare_winograd \
    --standard-checkpoint standard_model.ckpt \
    --winograd-checkpoint winograd_model.ckpt
```

## ⚡ Quick Start (Recommended)

**Option 1: Full Training (2-3 hours)**
```bash
cd "/Users/ishaanupponi/Documents/My projects/Ml/MACHINE LEARNING FROM UROP"
./train_all_models.sh 50 128
```

**Option 2: Quick Test (5 minutes)**
```bash
cd "/Users/ishaanupponi/Documents/My projects/Ml/MACHINE LEARNING FROM UROP"
python3 quick_comparison.py
```

## 📈 Expected Results

| **Model** | **Parameters** | **MACs** | **Accuracy** | **Training Time** |
|-----------|----------------|----------|--------------|-------------------|
| **ResNet-18** | 11.2M | 555M | 75-80% | ~2 hours |
| **Standard CNN** | 258K | 15.8M | 65-75% | ~30 minutes |
| **Winograd CNN** | 258K | 7.0M | 65-75% | ~30 minutes |

## 🔧 Troubleshooting

### Common Issues:

1. **Python not found**: Use `python3` instead of `python`
2. **Module not found**: Make sure you're in the correct directory
3. **CUDA out of memory**: Reduce batch size (e.g., `--batch-size 64`)
4. **Permission denied**: Run `chmod +x train_all_models.sh`

### Dependencies:
```bash
# Install required packages
pip install torch torchvision torchaudio
pip install numpy matplotlib
```

## 📁 Output Files

After training, you'll find:
- **ResNet-18**: `ResNet-18 inference/resnet18_final.ckpt`
- **Standard CNN**: `Initial our proposed model/lightweight_cnn_project/standard_final.ckpt`
- **Winograd CNN**: `Initial our proposed model/lightweight_cnn_project/winograd_final.ckpt`

## 🎉 Final Comparison

The scripts will automatically generate:
- ✅ Parameter counts and model sizes
- ✅ MAC (computational) comparisons  
- ✅ Training accuracy results
- ✅ Energy efficiency metrics
- ✅ Comprehensive performance analysis

**Key Finding**: Winograd CNN achieves **54% MAC reduction** compared to standard CNN while maintaining the same accuracy!
