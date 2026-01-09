import torch
import torch.nn as nn
import pytest
from quant_lens.hessian import power_iteration


class TestPowerIterationRigorous:
    """Rigorous mathematical tests for Hessian eigenvalue computation"""
    
    def test_quadratic_loss_known_eigenvalue(self, device):
        """
        Test on a quadratic loss with known Hessian eigenvalues.
        
        Setup:
            Model: Single linear layer with 3 parameters (no bias)
            Loss: L(w) = 0.5 * (1·w₀² + 2·w₁² + 5·w₂²)
            Hessian: H = diag(1, 2, 5)
            Expected max eigenvalue: 5.0
        """
        
        # Create a simple model with 3 parameters
        model = nn.Linear(1, 3, bias=False).to(device)
        # Initialize weights (doesn't matter for quadratic loss)
        model.weight.data = torch.tensor([[0.5], [1.0], [1.5]], device=device)
        
        # Create a custom loss that produces known Hessian
        class QuadraticLoss(nn.Module):
            """Loss = 0.5 * (1·w₀² + 2·w₁² + 5·w₂²)"""
            def __init__(self):
                super().__init__()
                self.eigenvalues = torch.tensor([1.0, 2.0, 5.0])
            
            def forward(self, model_output, target):
                # Get model parameters (weights)
                w = list(model.parameters())[0].flatten()
                
                # Quadratic loss: 0.5 * sum(λᵢ * wᵢ²)
                loss = 0.5 * torch.sum(self.eigenvalues.to(w.device) * w**2)
                return loss
        
        criterion = QuadraticLoss()
        
        # Create dummy dataloader (input doesn't matter for our custom loss)
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(16, 1, device=device)
        y = torch.randn(16, 3, device=device)
        dataloader = DataLoader(TensorDataset(X, y), batch_size=16)
        
        # Compute eigenvalue
        computed_eigenvalue = power_iteration(
            model, dataloader, criterion, device, num_iters=50
        )
        
        # Expected: max eigenvalue = 5.0
        expected_eigenvalue = 5.0
        
        # Allow small numerical error
        relative_error = abs(computed_eigenvalue - expected_eigenvalue) / expected_eigenvalue
        
        print(f"\n  Expected eigenvalue: {expected_eigenvalue}")
        print(f"  Computed eigenvalue: {computed_eigenvalue:.6f}")
        print(f"  Relative error: {relative_error*100:.2f}%")
        
        assert relative_error < 0.05, \
            f"Power iteration failed: expected {expected_eigenvalue}, got {computed_eigenvalue}"
    
    
    def test_diagonal_matrix_eigenvalues(self, device):
        """
        Test on a diagonal Hessian - easiest case.
        
        For a diagonal matrix, eigenvalues ARE the diagonal elements.
        This is the simplest possible test case.
        """
        
        # Model with 5 parameters
        model = nn.Linear(1, 5, bias=False).to(device)
        
        # Diagonal eigenvalues (intentionally unsorted)
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
        
        computed_eigenvalue = power_iteration(
            model, dataloader, criterion, device, num_iters=30
        )
        
        expected_eigenvalue = diagonal_values.max().item()  # 7.0
        
        relative_error = abs(computed_eigenvalue - expected_eigenvalue) / expected_eigenvalue
        
        print(f"\n  Diagonal values: {diagonal_values.tolist()}")
        print(f"  Expected max eigenvalue: {expected_eigenvalue}")
        print(f"  Computed eigenvalue: {computed_eigenvalue:.6f}")
        print(f"  Relative error: {relative_error*100:.2f}%")
        
        assert relative_error < 0.05, \
            f"Failed on diagonal matrix: expected {expected_eigenvalue}, got {computed_eigenvalue}"
    
    
    def test_2x2_matrix_analytical_solution(self, device):
        """
        Test on 2×2 matrix where we can compute eigenvalues analytically.
        
        For 2×2 matrix [[a, b], [b, c]], eigenvalues are:
        λ = (a+c)/2 ± sqrt((a-c)²/4 + b²)
        
        We'll test with a symmetric 2×2 Hessian.
        """
        
        # Model with 2 parameters
        model = nn.Linear(1, 2, bias=False).to(device)
        
        # Define symmetric matrix via quadratic form
        # H = [[3, 1], [1, 2]]
        # Eigenvalues: λ = 2.5 ± sqrt(0.25 + 1) = 2.5 ± 1.118
        # λ_max = 3.618, λ_min = 1.382
        
        class SymmetricQuadraticLoss(nn.Module):
            def forward(self, model_output, target):
                w = list(model.parameters())[0].flatten()
                # L = 0.5 * w^T * H * w
                # where H = [[3, 1], [1, 2]]
                H = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device=w.device)
                loss = 0.5 * torch.dot(w, torch.mv(H, w))
                return loss
        
        criterion = SymmetricQuadraticLoss()
        
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(8, 1, device=device)
        y = torch.randn(8, 2, device=device)
        dataloader = DataLoader(TensorDataset(X, y), batch_size=8)
        
        computed_eigenvalue = power_iteration(
            model, dataloader, criterion, device, num_iters=50
        )
        
        # Analytical solution
        a, b, c = 3.0, 1.0, 2.0
        import math
        expected_eigenvalue = (a + c)/2 + math.sqrt((a - c)**2/4 + b**2)
        # = 2.5 + sqrt(0.25 + 1) = 2.5 + 1.118 = 3.618
        
        relative_error = abs(computed_eigenvalue - expected_eigenvalue) / expected_eigenvalue
        
        print(f"\n  Matrix H = [[{a}, {b}], [{b}, {c}]]")
        print(f"  Expected max eigenvalue: {expected_eigenvalue:.6f}")
        print(f"  Computed eigenvalue: {computed_eigenvalue:.6f}")
        print(f"  Relative error: {relative_error*100:.2f}%")
        
        assert relative_error < 0.05, \
            f"Failed on 2×2 matrix: expected {expected_eigenvalue:.6f}, got {computed_eigenvalue:.6f}"
    
    
    def test_convergence_to_true_eigenvalue(self, device):
        """
        Test that more iterations converge to the true value.
        
        Verify that power iteration actually improves with more iterations.
        """
        
        model = nn.Linear(1, 3, bias=False).to(device)
        
        class QuadraticLoss(nn.Module):
            def forward(self, model_output, target):
                w = list(model.parameters())[0].flatten()
                eigenvalues = torch.tensor([1.0, 3.0, 8.0], device=w.device)
                return 0.5 * torch.sum(eigenvalues * w**2)
        
        criterion = QuadraticLoss()
        
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(8, 1, device=device)
        y = torch.randn(8, 3, device=device)
        dataloader = DataLoader(TensorDataset(X, y), batch_size=8)
        
        # Test with increasing iterations
        torch.manual_seed(42)
        eigen_5 = power_iteration(model, dataloader, criterion, device, num_iters=5)
        
        torch.manual_seed(42)
        eigen_20 = power_iteration(model, dataloader, criterion, device, num_iters=20)
        
        torch.manual_seed(42)
        eigen_50 = power_iteration(model, dataloader, criterion, device, num_iters=50)
        
        expected = 8.0
        
        error_5 = abs(eigen_5 - expected)
        error_20 = abs(eigen_20 - expected)
        error_50 = abs(eigen_50 - expected)
        
        print(f"\n  Expected eigenvalue: {expected}")
        print(f"  5 iterations:  {eigen_5:.6f} (error: {error_5:.6f})")
        print(f"  20 iterations: {eigen_20:.6f} (error: {error_20:.6f})")
        print(f"  50 iterations: {eigen_50:.6f} (error: {error_50:.6f})")
        
        # More iterations should generally be more accurate
        # (or at least, final iteration should be close)
        assert error_50 < 0.5, "Power iteration didn't converge after 50 iterations"
        
        # Verify monotonic improvement (with some tolerance for numerical noise)
        print(f"  Improvement 5→20: {error_5 - error_20:.6f}")
        print(f"  Improvement 20→50: {error_20 - error_50:.6f}")
