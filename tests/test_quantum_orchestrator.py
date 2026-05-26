"""
Unit Tests for Quantum Neural Orchestrator

This module contains tests for:
- Quantum-inspired neural network components
- Data processing pipeline
- Orchestration system
"""

import numpy as np
import pytest
import tensorflow as tf
import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quantum_core.qnn_module import QuantumInspiredDense, QuantumFeatureMap, build_hybrid_model
from src.ml_engines.data_processor import DataProcessor
from src.orchestration_layer.orchestrator import Orchestrator, AgentStatus

# Test QuantumInspiredDense layer
class TestQuantumInspiredDense:
    def test_layer_creation(self):
        layer = QuantumInspiredDense(units=10)
        assert layer.units == 10

    def test_layer_output_shape(self):
        layer = QuantumInspiredDense(units=10)
        input_data = tf.random.normal((5, 8))
        layer.build(input_data.shape)
        output = layer(input_data)
        assert output.shape == (5, 10)

    def test_complex_weights(self):
        layer = QuantumInspiredDense(units=10)
        input_data = tf.random.normal((5, 8))
        layer.build(input_data.shape)
        
        # Check that complex weights are created
        assert layer.w_real.shape == (8, 10)
        assert layer.w_imag.shape == (8, 10)
        assert layer.b.shape == (10,)

# Test QuantumFeatureMap layer
class TestQuantumFeatureMap:
    def test_feature_map_creation(self):
        feature_map = QuantumFeatureMap(features=4)
        assert feature_map.features == 4

    def test_feature_map_output_shape(self):
        feature_map = QuantumFeatureMap(features=4)
        input_data = tf.random.normal((5, 8))
        feature_map.build(input_data.shape)
        output = feature_map(input_data)
        assert output.shape == (5, 8)  # 4 sin + 4 cos = 8 features

# Test hybrid model
class TestHybridModel:
    def test_model_creation(self):
        model = build_hybrid_model(input_shape=(10,), num_classes=3)
        assert model.input_shape == (None, 10)
        assert model.output_shape == (None, 3)

    def test_model_training(self):
        model = build_hybrid_model(input_shape=(10,), num_classes=3)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 3, size=100)
        
        history = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
        assert 'loss' in history.history
        assert 'accuracy' in history.history

# Test DataProcessor
class TestDataProcessor:
    def test_data_loading(self, tmp_path):
        # Create a test CSV file
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2,target\n1,2,0\n3,4,1\n5,6,0")
        
        processor = DataProcessor()
        data = processor.load_data(str(test_file))
        assert data.shape == (3, 3)
        assert list(data.columns) == ['col1', 'col2', 'target']

    def test_data_preprocessing(self):
        data = [[1, 2, 0], [3, 4, 1], [5, 6, 0]]
        df = pd.DataFrame(data, columns=['col1', 'col2', 'target'])
        
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.preprocess(df, target_column='target')
        
        assert X_train.shape == (2, 2)
        assert X_test.shape == (1, 2)
        assert len(y_train) == 2
        assert len(y_test) == 1

# Test Orchestrator
class TestOrchestrator:
    def test_orchestrator_creation(self):
        orchestrator = Orchestrator()
        assert orchestrator.system_health == 1.0
        assert len(orchestrator.agents) == 0

    def test_agent_management(self):
        orchestrator = Orchestrator()
        
        # Add an agent
        orchestrator.add_agent("test_agent", ["test_capability"])
        assert "test_agent" in orchestrator.agents
        assert orchestrator.agents["test_agent"].status == AgentStatus.IDLE
        
        # Remove an agent
        orchestrator.remove_agent("test_agent")
        assert "test_agent" not in orchestrator.agents

    def test_task_submission(self):
        orchestrator = Orchestrator()
        orchestrator.add_agent("test_agent", ["test_capability"])
        
        # Submit a task
        orchestrator.submit_task("test_task", priority=1)
        assert orchestrator.task_queue.qsize() == 0  # Task should be assigned
        assert orchestrator.agents["test_agent"].current_task == "test_task"

if __name__ == "__main__":
    pytest.main([__file__])