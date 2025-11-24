#!/usr/bin/env python3
"""
Direct Energy Measurement for Jetson Nano
Measures actual power consumption during model inference
"""

import subprocess
import time
import re
import torch
import numpy as np
from pathlib import Path

class JetsonEnergyMonitor:
    """Monitor Jetson Nano power consumption using tegrastats."""

    def __init__(self, measurement_duration=10):
        self.measurement_duration = measurement_duration
        self.power_readings = []

    def _parse_tegrastats_output(self, output):
        """Parse tegrastats output to extract power values."""
        # Extract power consumption (format: "PWR 1234/1234" where first number is current power)
        power_match = re.search(r'PWR (\d+)/(\d+)', output)
        if power_match:
            current_power = int(power_match.group(1))  # Current power in mW
            return current_power / 1000  # Convert to Watts
        return None

    def measure_power(self, duration=None):
        """Measure power consumption for specified duration."""
        if duration is None:
            duration = self.measurement_duration

        self.power_readings = []
        start_time = time.time()

        try:
            # Start tegrastats process
            process = subprocess.Popen(
                ['tegrastats'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )

            while time.time() - start_time < duration:
                # Read output line
                output = process.stdout.readline()
                if output:
                    power = self._parse_tegrastats_output(output)
                    if power is not None:
                        timestamp = time.time() - start_time
                        self.power_readings.append((timestamp, power))

                time.sleep(0.1)  # Sample every 100ms

            process.terminate()
            process.wait()

        except KeyboardInterrupt:
            process.terminate()
            process.wait()

        return self.power_readings

    def get_average_power(self):
        """Calculate average power consumption."""
        if not self.power_readings:
            return 0.0

        powers = [reading[1] for reading in self.power_readings]
        return np.mean(powers)

    def get_power_stats(self):
        """Get comprehensive power statistics."""
        if not self.power_readings:
            return None

        powers = np.array([reading[1] for reading in self.power_readings])

        return {
            'average_power_w': np.mean(powers),
            'max_power_w': np.max(powers),
            'min_power_w': np.min(powers),
            'std_power_w': np.std(powers),
            'total_energy_j': np.mean(powers) * self.measurement_duration
        }

def measure_model_energy(model, test_loader, model_name, device, num_batches=50):
    """Measure energy consumption during model inference."""

    monitor = JetsonEnergyMonitor(measurement_duration=30)

    print(f"🔋 Measuring energy consumption for {model_name}...")
    print("Warming up...")

    # Warmup run
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= 5:  # Short warmup
                break
            inputs = inputs.to(device)
            _ = model(inputs)

    print("Starting energy measurement...")

    # Actual measurement
    power_readings = monitor.measure_power()

    if not power_readings:
        print("❌ Failed to collect power readings")
        return None

    # Run inference during measurement
    model.eval()
    inference_times = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= num_batches:
                break

            inputs = inputs.to(device)

            start_time = time.time()
            _ = model(inputs)
            torch.cuda.synchronize()  # Ensure GPU operations complete
            end_time = time.time()

            inference_times.append(end_time - start_time)

    # Calculate energy statistics
    stats = monitor.get_power_stats()
    avg_inference_time = np.mean(inference_times)
    energy_per_image = (stats['average_power_w'] * avg_inference_time) / num_batches

    print(f"📊 {model_name} Energy Results:")
    print(f"  Average Power: {stats['average_power_w']:.3f} W")
    print(f"  Max Power: {stats['max_power_w']:.3f} W")
    print(f"  Energy per Image: {energy_per_image*1000:.2f} mJ")
    print(f"  Average Inference Time: {avg_inference_time*1000:.2f} ms")

    return {
        'model_name': model_name,
        'avg_power_w': stats['average_power_w'],
        'max_power_w': stats['max_power_w'],
        'energy_per_image_j': energy_per_image,
        'avg_inference_time_s': avg_inference_time,
        'total_energy_j': stats['total_energy_j']
    }

def create_energy_comparison_plot(results):
    """Create visualization comparing energy consumption across models."""
    import matplotlib.pyplot as plt

    models = [r['model_name'] for r in results]
    energies = [r['energy_per_image_j'] * 1000 for r in results]  # Convert to mJ

    plt.figure(figsize=(12, 8))

    # Bar plot of energy consumption
    bars = plt.bar(models, energies, color=['blue', 'green', 'red'])
    plt.ylabel('Energy per Image (mJ)')
    plt.title('Energy Consumption Comparison: Model Inference on Jetson Nano')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{energy:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('jetson_energy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return 'jetson_energy_comparison.png'

if __name__ == "__main__":
    print("🔋 Jetson Nano Energy Measurement Tool")
    print("=" * 50)

    # Example usage would go here
    # This would integrate with your existing model comparison scripts
