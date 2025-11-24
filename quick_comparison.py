#!/usr/bin/env python3
"""
Quick comparison script to test all three models without full training.
This script loads existing checkpoints or runs short training for comparison.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_quick_test():
    """Run quick tests of all models."""
    base_dir = Path("/Users/ishaanupponi/Documents/My projects/Ml/MACHINE LEARNING FROM UROP")
    resnet_dir = base_dir / "ResNet-18 inference"
    lightweight_dir = base_dir / "Initial our proposed model" / "lightweight_cnn_project"
    
    print("🚀 QUICK MODEL COMPARISON")
    print("=" * 60)
    
    # Test 1: Model Architecture Analysis
    print("\n📊 STEP 1: MODEL ARCHITECTURE ANALYSIS")
    print("-" * 40)
    
    # Standard CNN
    print("\n🔍 Standard Lightweight CNN:")
    try:
        os.chdir(lightweight_dir)
        result = subprocess.run(
            ["python3", "-m", "src.lightweight_cnn.model_standard"],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error testing standard model: {e}")
    
    # Winograd CNN
    print("\n🔍 Winograd Lightweight CNN:")
    try:
        os.chdir(lightweight_dir)
        result = subprocess.run(
            ["python3", "-m", "src.lightweight_cnn.model_winograd"],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error testing Winograd model: {e}")
    
    # Test 2: Quick Training (5 epochs each)
    print("\n🏋️ STEP 2: QUICK TRAINING TEST (5 epochs each)")
    print("-" * 40)
    
    models_to_train = [
        ("Standard CNN", "python3 -m src.scripts.train_standard --epochs 5 --batch-size 64 --save-path quick_standard.ckpt"),
        ("Winograd CNN", "python3 -m src.scripts.train_winograd --epochs 5 --batch-size 64 --save-path quick_winograd.ckpt")
    ]
    
    for model_name, cmd in models_to_train:
        print(f"\n🚀 Quick training {model_name}...")
        try:
            os.chdir(lightweight_dir)
            subprocess.run(cmd.split(), check=True)
            print(f"✅ {model_name} quick training completed!")
        except Exception as e:
            print(f"❌ {model_name} quick training failed: {e}")
    
    # Test 3: Comparison
    print("\n⚖️ STEP 3: MODEL COMPARISON")
    print("-" * 40)
    
    try:
        os.chdir(lightweight_dir)
        subprocess.run([
            "python3", "-m", "src.scripts.compare_winograd",
            "--standard-checkpoint", "quick_standard.ckpt",
            "--winograd-checkpoint", "quick_winograd.ckpt"
        ], check=True)
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
    
    print(f"\n🎉 QUICK COMPARISON COMPLETED!")

if __name__ == "__main__":
    run_quick_test()
