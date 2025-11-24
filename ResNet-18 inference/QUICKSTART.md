# Quick Start Guide - ResNet-18 for CIFAR-100

## Installation

```bash
cd "ResNet-18 inference"
pip install -r requirements.txt
```

## Quick Commands

### 1. View Model Statistics (No Training Required)
```bash
python3 -m src.scripts.measure_params_macs
```
This shows:
- Total parameters: **11,220,132** (43.4x more than Lightweight CNN)
- Total MACs: **555,468,800** (35.3x more than Lightweight CNN)
- Model size: **42.80 MB**
- Detailed comparison with Lightweight CNN

### 2. Train ResNet-18 on CIFAR-100
```bash
# Basic training (100 epochs, ~2-3 hours on GPU)
python3 -m src.scripts.train --epochs 100 --batch-size 128

# Quick test (10 epochs, ~15-20 minutes on GPU)
python3 -m src.scripts.train --epochs 10 --batch-size 128

# Extended training for better accuracy (200 epochs)
python3 -m src.scripts.train --epochs 200 --batch-size 128 --lr 0.1
```

**Expected Results:**
- After 100 epochs: ~75-78% accuracy
- After 200 epochs: ~78-80% accuracy
- Training time (GPU): ~1.5-2 minutes per epoch
- Training time (CPU): ~15-20 minutes per epoch

### 3. Evaluate Trained Model
```bash
python3 -m src.scripts.evaluate --checkpoint best_resnet18.ckpt
```

### 4. Compare with Lightweight CNN
```bash
python3 -m src.scripts.compare_models \
    --resnet-checkpoint best_resnet18.ckpt \
    --lightweight-checkpoint "../Initial our proposed model/lightweight_cnn_project/best_model.ckpt"
```

This provides comprehensive comparison:
- Accuracy comparison (Top-1 and Top-5)
- Inference speed comparison
- Energy efficiency metrics
- Use case recommendations

## Expected Performance

| Metric | Lightweight CNN | ResNet-18 | Advantage |
|--------|----------------|-----------|-----------|
| **Parameters** | 258K | 11.2M | 43.4x smaller (Light) |
| **MACs** | 15.8M | 555M | 35.3x fewer (Light) |
| **Model Size** | 0.99 MB | 42.8 MB | 43.4x smaller (Light) |
| **Top-1 Accuracy** | ~65-75% | ~75-80% | +10-15% (ResNet) |
| **Inference Speed** | ~2-5 ms | ~10-25 ms | 5-10x faster (Light) |
| **Energy Efficiency** | High | Low | Much better (Light) |

## Training Tips

1. **GPU Recommended**: Training on CPU is very slow (~15-20 min/epoch)
2. **Batch Size**: Use 128 for most GPUs, increase to 256 if you have >8GB VRAM
3. **Learning Rate**: Default 0.1 with cosine annealing works well
4. **Epochs**: 100 epochs minimum, 200 for best results
5. **Data Augmentation**: Already included (random crop, horizontal flip)

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python3 -m src.scripts.train --epochs 100 --batch-size 64
```

### Slow Training
- Use GPU if available (CUDA)
- Reduce number of workers if CPU is bottleneck
- Consider training fewer epochs for quick testing

### Resume Training
```bash
python3 -m src.scripts.train --resume best_resnet18.ckpt --epochs 200
```

## Key Insights

✓ **ResNet-18 is better when:**
- Maximum accuracy is required
- GPU resources are available
- Server-side inference
- Accuracy > efficiency

✓ **Lightweight CNN is better when:**
- Deploying to mobile/edge devices
- Energy efficiency is critical
- Real-time inference required
- Limited computational resources
- Battery-powered devices

## Next Steps

1. Train both models on CIFAR-100
2. Compare their performance using `compare_models.py`
3. Analyze the accuracy vs efficiency trade-off
4. Choose the appropriate model for your use case
