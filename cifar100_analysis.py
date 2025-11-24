import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.optimizers import Adam
import time

# Load CIFAR-100 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Function to create a CNN model with variable kernel counts
def create_model(kernel_counts):
    model = models.Sequential()
    model.add(layers.Conv2D(kernel_counts[0], (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    for kernels in kernel_counts[1:]:
        model.add(layers.Conv2D(kernels, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(100))  # 100 classes in CIFAR-100
    
    return model

# Function to calculate arithmetic complexity (FLOPs)
def calculate_flops(model):
    # This is a simplified calculation
    flops = 0
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            # FLOPs = output_h * output_w * kernel_h * kernel_w * in_channels * out_channels
            output_shape = layer.output_shape
            kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
            in_channels = layer.input_shape[-1]
            out_channels = layer.output_shape[-1]
            flops += np.prod(output_shape[1:-1]) * kernel_size * in_channels * out_channels
    return flops

# Experiment with different kernel counts
kernel_configs = [
    [32],
    [32, 64],
    [32, 64, 128],
    [64, 128, 256],
    [64, 128, 256, 512]
]

results = []

for i, kernels in enumerate(kernel_configs):
    print(f"\nTraining model with kernel counts: {kernels}")
    
    # Create and compile model
    model = create_model(kernels)
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    
    # Calculate complexity
    flops = calculate_flops(model)
    param_count = model.count_params()
    
    # Train model
    start_time = time.time()
    history = model.fit(train_images, train_labels, epochs=5, 
                       validation_data=(test_images, test_labels),
                       batch_size=64, verbose=1)
    training_time = time.time() - start_time
    
    # Get final accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    
    # Store results
    results.append({
        'kernel_counts': kernels,
        'num_kernels': len(kernels),
        'flops': flops,
        'parameters': param_count,
        'test_accuracy': test_acc,
        'training_time': training_time
    })
    
    print(f"Kernel counts: {kernels}, Test accuracy: {test_acc:.4f}, FLOPs: {flops:.2e}, Params: {param_count:,}")

# Plot results
plt.figure(figsize=(12, 5))

# Plot 1: Test Accuracy vs Number of Kernels
plt.subplot(1, 2, 1)
plt.plot([r['num_kernels'] for r in results], [r['test_accuracy'] for r in results], 'o-')
plt.xlabel('Number of Convolutional Layers')
plt.ylabel('Test Accuracy')
plt.title('Model Performance vs Architecture Complexity')
plt.grid(True)

# Plot 2: Arithmetic Complexity (FLOPs) vs Number of Kernels
plt.subplot(1, 2, 2)
plt.plot([r['num_kernels'] for r in results], [r['flops']/1e6 for r in results], 's-r')
plt.xlabel('Number of Convolutional Layers')
plt.ylabel('Arithmetic Complexity (MFLOPs)')
plt.title('Arithmetic Complexity vs Architecture')
plt.grid(True)

plt.tight_layout()
plt.savefig('complexity_analysis.png')
plt.show()

# Print results summary
print("\nSummary of Results:")
print("-" * 80)
print(f"{'Kernel Counts':<20} {'Layers':<10} {'FLOPs (M)':<15} {'Params':<15} {'Test Acc':<10} {'Train Time (s)'}")
print("-" * 80)
for r in results:
    print(f"{str(r['kernel_counts']):<20} {r['num_kernels']:<10} {r['flops']/1e6:<15.2f} {r['parameters']:<15,} {r['test_accuracy']:<10.4f} {r['training_time']:.1f}")
