# Lightweight CNN Project for CIFAR-100

This project provides a small, easy-to-understand PyTorch implementation of a lightweight Convolutional Neural Network (CNN) for image classification on the CIFAR-100 dataset.

The code is intentionally simple, well-commented, and organized into logical modules to be accessible for students, researchers, and anyone new to PyTorch.

## Project Goals

- Implement a simple CNN architecture with a clear stem, core, and head structure.
- Organize each logical layer into its own Python file for clarity.
- Provide simple, standalone scripts for training, evaluation, model export, and analysis.
- Use standard Python and PyTorch practices, with minimal external dependencies.

## Code Organization

The project follows a standard Python package structure:

```
lightweight_cnn_project/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ lightweight_cnn/       # Main package source code
│  │  ├─ __init__.py
│  │  ├─ model.py           # Assembles the full CNN model
│  │  ├─ layers/            # Individual nn.Module layers
│  │  ├─ utils.py           # Helper functions (e.g., parameter counting)
│  │  └─ macs.py            # MACs computation utility
│  ├─ scripts/             # Executable scripts
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  ├─ export_onnx.py
│  │  └─ measure_params_macs.py
│  └─ tests/               # Unit tests
│     ├─ test_forward_shape.py
│     └─ test_count_params.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd lightweight_cnn_project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

All scripts are run as modules from the project's root directory.

### 1. Measure Model Parameters and MACs

This script builds the model and prints a table of parameters and Multiply-Accumulate operations (MACs) per layer.

```bash
python -m src.scripts.measure_params_macs
```

### 2. Train the Model

This script trains the Lightweight CNN on the CIFAR-100 dataset. The best model checkpoint will be saved to `best_model.ckpt`.

```bash
# Train for 100 epochs with a batch size of 128
python -m src.scripts.train --epochs 100 --batch-size 128

# Train for a custom number of epochs
python -m src.scripts.train --epochs 50
```

### 3. Evaluate the Model

Evaluate the trained model on the CIFAR-100 test set.

```bash
# Evaluate the best model saved from training
python -m src.scripts.evaluate --checkpoint best_model.ckpt
```

### 4. Export the Model

Export the model to ONNX and TorchScript formats for deployment.

```bash
# Export a trained model
python -m src.scripts.export_onnx --checkpoint best_model.ckpt

# Export a randomly initialized model (if no checkpoint is available)
python -m src.scripts.export_onnx
```

### 5. Run Tests

Run the unit tests using `pytest` to ensure the model forward pass and utilities are working correctly.

```bash
pytest
```
