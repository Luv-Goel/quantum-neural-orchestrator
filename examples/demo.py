"""
Quick demo script showing Quantum Neural Orchestrator in action

This example demonstrates:
1. Creating synthetic data
2. Training a hybrid quantum-classical model
3. Using the orchestration system to manage the workflow
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.ml_engines.data_processor import DataProcessor
from src.quantum_core.qnn_module import build_hybrid_model
from src.orchestration_layer.orchestrator import Orchestrator

def main():
    print("=" * 60)
    print("Quantum Neural Orchestrator - Quick Demo")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n[1] Generating synthetic dataset...")
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    n_classes = 3
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    print(f"   Created {n_samples} samples with {n_features} features, {n_classes} classes")
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Step 2: Build and train quantum-inspired model
    print("\n[2] Building hybrid quantum-classical model...")
    model = build_hybrid_model(input_shape=(n_features,), num_classes=n_classes)
    model.summary()
    
    print("\n[3] Training model for 5 epochs...")
    history = model.fit(X_train, y_train, 
                       epochs=5, 
                       batch_size=16, 
                       validation_data=(X_test, y_test),
                       verbose=1)
    
    # Step 3: Evaluate model
    print("\n[4] Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Test loss: {test_loss:.4f}")
    
    # Step 4: Demo orchestration
    print("\n[5] Demonstrating orchestration...")
    orchestrator = Orchestrator()
    
    # Add agents
    orchestrator.add_agent("model_trainer", ["ml_training"])
    orchestrator.add_agent("data_processor", ["data_ingestion"])
    
    # Submit tasks
    orchestrator.submit_task("validate_data", priority=1)
    orchestrator.submit_task("run_inference", priority=2)
    
    # Show status
    import json
    status = orchestrator.get_system_status()
    print(f"   System health: {status['system_health']:.2f}")
    print(f"   Active agents: {list(status['agents'].keys())}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()