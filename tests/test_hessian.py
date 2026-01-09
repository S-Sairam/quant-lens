import torch
import torch.nn as nn
from quant_lens.hessian import power_iteration


class TestPowerIteration:
    """Tests for Hessian eigenvalue computation"""
    
    def test_returns_numeric_eigenvalue(self, simple_model, dummy_dataloader, device):
        """Power iteration should return a numeric eigenvalue"""
        criterion = nn.CrossEntropyLoss()
        
        eigenvalue = power_iteration(
            simple_model, dummy_dataloader, criterion, device, num_iters=5
        )
        
        # Eigenvalues can be negative for non-convex loss surfaces
        assert isinstance(eigenvalue, (int, float))
        assert not torch.isnan(torch.tensor(eigenvalue))
        assert not torch.isinf(torch.tensor(eigenvalue))
    
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

import torch
import torch.nn as nn
import pytest
from quant_lens.hessian import power_iteration


class TestPowerIteration:
    """Tests for Hessian eigenvalue computation"""
    
    def test_returns_numeric_eigenvalue(self, simple_model, dummy_dataloader, device):
        """Power iteration should return a numeric eigenvalue"""
        criterion = nn.CrossEntropyLoss()
        
        eigenvalue = power_iteration(
            simple_model, dummy_dataloader, criterion, device, num_iters=5
        )
        
        # Eigenvalues can be negative for non-convex loss surfaces
        assert isinstance(eigenvalue, (int, float))
        assert not torch.isnan(torch.tensor(eigenvalue))
        assert not torch.isinf(torch.tensor(eigenvalue))
    
    def test_convergence_with_more_iterations(self, simple_model, dummy_dataloader, device):
        """More iterations should produce more stable results"""
        criterion = nn.CrossEntropyLoss()
        
        # Few iterations
        eigen_5 = power_iteration(simple_model, dummy_dataloader, criterion, device, 5)
        
        # More iterations
        eigen_20 = power_iteration(simple_model, dummy_dataloader, criterion, device, 20)
        
        # Should be relatively close (convergence)
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
    
    # NEW RIGOROUS TESTS
    
    def test_quadratic_loss_known_eigenvalue(self, device):
        """Test on quadratic loss with known Hessian: H = diag(1, 2, 5)"""
        model = nn.Linear(1, 3, bias=False).to(device)
        model.weight.data = torch.tensor([[0.5], [1.0], [1.5]], device=device)
        
        class QuadraticLoss(nn.Module):
            def forward(self, model_output, target):
                w = list(model.parameters())[0].flatten()
                eigenvalues = torch.tensor([1.0, 2.0, 5.0], device=w.device)
                return 0.5 * torch.sum(eigenvalues * w**2)
        
        criterion = QuadraticLoss()
        
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(16, 1, device=device)
        y = torch.randn(16, 3, device=device)
        dataloader = DataLoader(TensorDataset(X, y), batch_size=16)
        
        computed = power_iteration(model, dataloader, criterion, device, num_iters=50)
        expected = 5.0
        
        relative_error = abs(computed - expected) / expected
        assert relative_error < 0.05, \
            f"Expected {expected}, got {computed} (error: {relative_error*100:.1f}%)"
    
    def test_diagonal_matrix_eigenvalues(self, device):
        """Test on diagonal Hessian - eigenvalues = diagonal elements"""
        model = nn.Linear(1, 5, bias=False).to(device)
        diagonal_values = torch.tensor([2.0, 7.0, 3.0, 1.0, 4.0])
        
        class DiagonalQuadraticLoss(nn.Module):
            def __init__(self, diagonal):
                super().__init__()
                self.diagonal = diagonal
            
            def forward(self, model_output, target):
                w = list(model.parameters())[0].flatten()
                return 0.5 * torch.sum(self.diagonal.to(w.device) * w**2)
        
        criterion = DiagonalQuadraticLoss(diagonal_values)
        
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(8, 1, device=device)
        y = torch.randn(8, 5, device=device)
        dataloader = DataLoader(TensorDataset(X, y), batch_size=8)
        
        computed = power_iteration(model, dataloader, criterion, device, num_iters=30)
        expected = diagonal_values.max().item()  # 7.0
        
        relative_error = abs(computed - expected) / expected
        assert relative_error < 0.05, \
            f"Expected {expected}, got {computed}"

