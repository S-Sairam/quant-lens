import torch
import torch.nn as nn
import pytest
from quant_lens.quantization import FakeQuantOp, QuantLinear, replace_linear_layers


class TestFakeQuantOp:
    """Tests for FakeQuantOp with known behaviors"""
    
    def test_8bit_quantization_range(self):
        """8-bit quantization should clip to [-128, 127]"""
        x = torch.tensor([[-200.0, 0.0, 200.0]])
        quantized = FakeQuantOp.apply(x, 8)
        
        # Values should be within 8-bit range after scaling
        assert quantized.min() >= -128 * (x.abs().max() / 127)
        assert quantized.max() <= 127 * (x.abs().max() / 127)
    
    def test_quantization_is_deterministic(self):
        """Same input should always produce same output"""
        x = torch.tensor([[1.5, 2.7, 3.9]])
        
        result1 = FakeQuantOp.apply(x, 8)
        result2 = FakeQuantOp.apply(x, 8)
        
        assert torch.allclose(result1, result2)
    
    def test_zero_input(self):
        """Zero input should return zero"""
        x = torch.zeros(5, 10)
        quantized = FakeQuantOp.apply(x, 8)
        assert torch.allclose(quantized, torch.zeros_like(x))
    
    def test_gradient_flows_backward(self):
        """STE should allow gradients to flow (not be zero)"""
        x = torch.tensor([[1.5, 2.7, 3.9]], requires_grad=True)
        y = FakeQuantOp.apply(x, 8)
        loss = y.sum()
        loss.backward()
        
        # Gradient should exist and not be all zeros
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    @pytest.mark.parametrize("num_bits,expected_levels", [
        (8, 256),   # 2^8 = 256 levels
        (4, 16),    # 2^4 = 16 levels
        (2, 4),     # 2^2 = 4 levels
    ])
    def test_different_bit_widths(self, num_bits, expected_levels):
        """Different bit widths should produce appropriate quantization"""
        x = torch.linspace(-10, 10, 1000)
        quantized = FakeQuantOp.apply(x, num_bits)
        
        # Count unique values (should be <= expected levels)
        unique_values = torch.unique(quantized)
        assert len(unique_values) <= expected_levels


class TestQuantLinear:
    """Tests for QuantLinear layer"""
    
    def test_output_shape(self):
        """QuantLinear should preserve output shape"""
        layer = QuantLinear(10, 5, num_bits=8)
        x = torch.randn(3, 10)
        output = layer(x)
        
        assert output.shape == (3, 5)
    
    def test_forward_pass_runs(self):
        """Forward pass should complete without errors"""
        layer = QuantLinear(10, 5, bias=True, num_bits=8)
        x = torch.randn(2, 10)
        
        try:
            output = layer(x)
            assert True
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
    
    def test_quantized_different_from_fp32(self):
        """Quantized output should differ from FP32 (due to quantization)"""
        # Create FP32 layer
        fp32_layer = nn.Linear(10, 5, bias=False)
        
        # Create QuantLinear with same weights
        quant_layer = QuantLinear(10, 5, bias=False, num_bits=8)
        quant_layer.weight.data = fp32_layer.weight.data.clone()
        
        x = torch.randn(2, 10)
        fp32_output = fp32_layer(x)
        quant_output = quant_layer(x)
        
        # Outputs should be different (quantization introduces error)
        assert not torch.allclose(fp32_output, quant_output, atol=1e-6)


class TestReplaceLinearLayers:
    """Tests for model conversion"""
    
    def test_converts_all_linear_layers(self, simple_model):
        """All nn.Linear layers should be replaced with QuantLinear"""
        quantized_model = replace_linear_layers(simple_model)
        
        linear_count = 0
        quant_linear_count = 0
        
        for module in quantized_model.modules():
            if isinstance(module, nn.Linear) and not isinstance(module, QuantLinear):
                linear_count += 1
            if isinstance(module, QuantLinear):
                quant_linear_count += 1
        
        # Should have 0 regular Linear, 2 QuantLinear (from simple_model)
        assert linear_count == 0
        assert quant_linear_count == 2
    
    def test_preserves_weights(self, simple_model):
        """Weight values should be copied (not lost)"""
        original_weight = simple_model[0].weight.data.clone()
        quantized_model = replace_linear_layers(simple_model)
        
        # First layer weights should match
        assert torch.allclose(quantized_model[0].weight.data, original_weight)
    
    def test_preserves_bias(self):
        """Bias should be preserved if it exists"""
        model = nn.Sequential(nn.Linear(5, 3, bias=True))
        model[0].bias.data = torch.tensor([1.0, 2.0, 3.0])
        
        quantized_model = replace_linear_layers(model)
        
        assert torch.allclose(quantized_model[0].bias.data, 
                            torch.tensor([1.0, 2.0, 3.0]))
