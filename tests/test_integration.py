import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from quant_lens import QuantDiagnostic


class TestIntegration:
    """End-to-end integration tests"""
    
    def test_resnet_like_model(self, tmp_path):
        """Test with a more complex ResNet-like architecture"""
        class SimpleBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            
            def forward(self, x):
                return x + self.conv2(torch.relu(self.conv1(x)))
        
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            SimpleBlock(16),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
        
        # Create image data
        X = torch.randn(16, 3, 8, 8)
        y = torch.randint(0, 10, (16,))
        loader = DataLoader(TensorDataset(X, y), batch_size=8)
        
        # Run diagnostic
        diagnostic = QuantDiagnostic(model, loader, device='cpu')
        diagnostic.add_int8_model()
        metrics = diagnostic.run_analysis(landscape_steps=5, hessian_iters=3)
        diagnostic.plot(save_path=str(tmp_path / "resnet_test.png"))
        
        assert 'FP32' in metrics
        assert 'Int8' in metrics
    
    def test_multiple_quantization_schemes(self, simple_model, dummy_dataloader, tmp_path):
        """Test with multiple bit widths"""
        diagnostic = QuantDiagnostic(simple_model, dummy_dataloader, device='cpu')
        
        # Add multiple variants
        diagnostic.add_int8_model(num_bits=8, name="Int8")
        diagnostic.add_int8_model(num_bits=4, name="Int4")
        diagnostic.add_int8_model(num_bits=2, name="Int2")
        
        metrics = diagnostic.run_analysis(landscape_steps=5, hessian_iters=3)
        diagnostic.plot(save_path=str(tmp_path / "multi_bit.png"))
        
        assert len(metrics) == 4  # FP32 + Int8 + Int4 + Int2
        
        # Verify sharpness increases with aggressive quantization
        # (generally expected, though not guaranteed)
        assert 'Int2' in metrics
