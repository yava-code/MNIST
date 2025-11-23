import torch
import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import MnistModel, MnistMLP, MnistCNN


class TestModelArchitectures:
    """Test that model architectures are correctly defined"""
    
    def test_logistic_model_creation(self):
        """Test that logistic regression model can be created"""
        model = MnistModel()
        assert model is not None
        assert hasattr(model, 'linear')
    
    def test_mlp_model_creation(self):
        """Test that MLP model can be created"""
        model = MnistMLP()
        assert model is not None
        assert hasattr(model, 'network')
    
    def test_cnn_model_creation(self):
        """Test that CNN model can be created"""
        model = MnistCNN()
        assert model is not None
        assert hasattr(model, 'network')
    
    def test_logistic_forward_pass(self):
        """Test forward pass through logistic model"""
        model = MnistModel()
        # Create a random input tensor (batch_size=1, channels=1, height=28, width=28)
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        # Output should be (batch_size=1, num_classes=10)
        assert output.shape == (1, 10)
    
    def test_mlp_forward_pass(self):
        """Test forward pass through MLP model"""
        model = MnistMLP()
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10)
    
    def test_cnn_forward_pass(self):
        """Test forward pass through CNN model"""
        model = MnistCNN()
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10)
    
    def test_logistic_output_range(self):
        """Test that model outputs reasonable values"""
        model = MnistModel()
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        # Logits can be any real number, just check they exist
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_models_in_eval_mode(self):
        """Test that models can be switched to eval mode"""
        models = [MnistModel(), MnistMLP(), MnistCNN()]
        for model in models:
            model.eval()
            assert not model.training


class TestModelLoading:
    """Test that saved models can be loaded"""
    
    def test_model_files_exist(self):
        """Test that model weight files exist"""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        assert os.path.exists(os.path.join(model_dir, 'mnist-logistic.pth'))
        assert os.path.exists(os.path.join(model_dir, 'mnist-mlp.pth'))
        assert os.path.exists(os.path.join(model_dir, 'mnist-cnn.pth'))
    
    def test_load_logistic_weights(self):
        """Test loading logistic regression weights"""
        model = MnistModel()
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mnist-logistic.pth')
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        # If loading succeeds without error, test passes
        assert True
    
    def test_load_mlp_weights(self):
        """Test loading MLP weights"""
        model = MnistMLP()
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mnist-mlp.pth')
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        assert True
    
    def test_load_cnn_weights(self):
        """Test loading CNN weights"""
        model = MnistCNN()
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mnist-cnn.pth')
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        assert True
