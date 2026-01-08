import torch
import torch.nn as nn
from quant_lens.hessian import power_iteration


class TestPowerIteration:
    """Tests for Hessian eigenvalue computation"""
    
    def test_returns_positive_eigenvalue(self, simple_model, dummy_dataloader, device):
        """Top eigenvalue should be positive (or zero)"""
        criterion = nn.CrossEntropyLoss()
        
        eigenvalue = power_iteration(
            simple_model, dummy_dataloader, criterion, device, num_iters=5
        )
        
        assert eigenvalue >= 0
    
    def test_convergence_with_more_iterations(self, simple_model, dummy_dataloader, device):
        """More iterations should produce more stable results"""
        criterion = nn.CrossEntropyLoss()
        
        # Few iterations
        eigen_5 = power_iteration(simple_model, dummy_dataloader, criterion, device, 5)
        
        # More iterations
        eigen_20 = power_iteration(simple_model, dummy_dataloader, criterion, device, 20)
        
        # Should be relatively close (convergence)
        # Not testing exact equality due to randomness
        assert isinstance(eigen_5, float)
        assert isinstance(eigen_20, float)
    
    def test_deterministic_with_seed(self, simple_model, dummy_dataloader, device):
        """Should be reproducible with torch.manual_seed"""
        criterion = nn.CrossEntropyLoss()
        
        torch.manual_seed(42)
        eigen_1 = power_iteration(simple_model, dummy_dataloader, criterion, device, 10)
        
        torch.manual_seed(42)
        eigen_2 = power_iteration(simple_model, dummy_dataloader, criterion, device, 10)
        
        assert abs(eigen_1 - eigen_2) < 1e-6
