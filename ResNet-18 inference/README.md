# ResNet-18 for CIFAR-100 Classification

This project provides a PyTorch implementation of ResNet-18 for CIFAR-100 image classification, designed for direct comparison with the Lightweight CNN model.

## Project Goals

- Implement standard ResNet-18 architecture adapted for CIFAR-100 (32×32 images)
- Provide training, evaluation, and analysis scripts
- Enable direct performance and efficiency comparison with Lightweight CNN
- Measure computational cost (parameters, MACs) and energy efficiency

## Model Architecture

ResNet-18 is a deep residual network with:
- **11.7M parameters** (45x more than Lightweight CNN)
- **~1.8B MACs** (114x more than Lightweight CNN)
- **18 layers** with skip connections
- **4 residual stages** with increasing channel dimensions

Architecture adapted for CIFAR-100:
- Initial 3×3 conv (no 7×7 conv or max pooling like ImageNet version)
- 4 residual stages: [64, 128, 256, 512] channels
- Global average pooling + FC layer for 100 classes

## Project Structure

```
ResNet-18 inference/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ resnet18/              # Model implementation
│  │  ├─ __init__.py
│  │  ├─ model.py            # ResNet-18 architecture
│  │  ├─ utils.py            # Helper functions
│  │  └─ macs.py             # MACs computation
│  └─ scripts/               # Executable scripts
│     ├─ train.py
│     ├─ evaluate.py
│     ├─ measure_params_macs.py
│     └─ compare_models.py   # Compare with Lightweight CNN
```

## Installation

1. Navigate to the project directory:
   ```bash
   cd "ResNet-18 inference"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Measure Model Parameters and MACs

```bash
python3 -m src.scripts.measure_params_macs
```

### 2. Train ResNet-18 on CIFAR-100

```bash
# Train for 100 epochs (default)
python3 -m src.scripts.train --epochs 100 --batch-size 128

# Train with custom settings
python3 -m src.scripts.train --epochs 200 --batch-size 256 --lr 0.1
```

### 3. Evaluate Trained Model

```bash
python3 -m src.scripts.evaluate --checkpoint best_resnet18.ckpt
```

### 4. Compare with Lightweight CNN

```bash
python3 -m src.scripts.compare_models \
    --resnet-checkpoint best_resnet18.ckpt \
    --lightweight-checkpoint ../Initial\ our\ proposed\ model/lightweight_cnn_project/best_model.ckpt
```

## Expected Performance

### Accuracy
- **ResNet-18**: 75-80% on CIFAR-100 test set
- **Training time**: 100-200 epochs for convergence

### Computational Cost
- **Parameters**: ~11.7M
- **MACs**: ~1.8B
- **Inference time (GPU)**: ~10-25ms per image
- **Model size**: ~45MB

### Comparison with Lightweight CNN
- **45x more parameters**
- **114x more MACs**
- **10-15% higher accuracy**
- **10-20x slower inference**
- **Much higher energy consumption**

## Training Tips

1. **Learning rate**: Start with 0.1, use cosine annealing or step decay
2. **Data augmentation**: Random crop, horizontal flip (already implemented)
3. **Regularization**: Weight decay (5e-4) helps prevent overfitting
4. **Batch size**: 128-256 works well on most GPUs
5. **Epochs**: 100-200 epochs for good convergence

## Citation

ResNet paper:
```
He, K., Zhang, X., Ren, S., & Sun, J. (2016).
Deep residual learning for image recognition.
In CVPR 2016.
```
