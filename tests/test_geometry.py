import torch
import numpy as np
from quant_lens.geometry import get_random_direction, trace_1d_loss
import torch.nn as nn

class TestGetRandomDirection:
    """Tests for random direction generation"""
    
    def test_returns_list_same_length(self, simple_model):
        """Should return one direction vector per parameter"""
        direction = get_random_direction(simple_model)
        param_count = sum(1 for p in simple_model.parameters() if p.requires_grad)
        
        assert len([d for d in direction if d is not None]) == param_count
    
    def test_filter_normalization_applied(self, known_weights_model):
        """Direction should be normalized to parameter norm"""
        direction = get_random_direction(known_weights_model)
        
        param = list(known_weights_model.parameters())[0]
        dir_vec = direction[0]
        
        # Direction norm should match parameter norm
        param_norm = param.data.norm().item()
        dir_norm = dir_vec.norm().item()
        
        assert abs(param_norm - dir_norm) < 1e-5
    
    def test_direction_is_random(self, simple_model):
        """Multiple calls should produce different directions"""
        dir1 = get_random_direction(simple_model)
        dir2 = get_random_direction(simple_model)
        
        # First parameter's direction should differ
        assert not torch.allclose(dir1[0], dir2[0])


class TestTrace1DLoss:
    """Tests for loss landscape tracing"""
    
    def test_returns_correct_number_of_points(self, simple_model, dummy_dataloader, device):
        """Should return exactly 'steps' number of points"""
        criterion = nn.CrossEntropyLoss()
        steps = 10
        
        alphas, losses = trace_1d_loss(
            simple_model, dummy_dataloader, criterion, device, steps=steps
        )
        
        assert len(alphas) == steps
        assert len(losses) == steps
    
    def test_alpha_range(self, simple_model, dummy_dataloader, device):
        """Alpha should range from -distance to +distance"""
        criterion = nn.CrossEntropyLoss()
        distance = 0.5
        
        alphas, _ = trace_1d_loss(
            simple_model, dummy_dataloader, criterion, device, 
            steps=10, distance=distance
        )
        
        assert abs(alphas[0] - (-distance)) < 1e-6
        assert abs(alphas[-1] - distance) < 1e-6
    
    def test_restores_original_weights(self, simple_model, dummy_dataloader, device):
        """Model weights should be restored after tracing"""
        criterion = nn.CrossEntropyLoss()
        original_weights = simple_model[0].weight.data.clone()
        
        # Trace landscape
        _, _ = trace_1d_loss(
            simple_model, dummy_dataloader, criterion, device, steps=5
        )
        
        # Weights should be unchanged
        assert torch.allclose(simple_model[0].weight.data, original_weights)
    
    def test_loss_at_origin_is_current_loss(self, simple_model, dummy_dataloader, device):
        """Loss at alpha=0 should match current model's loss"""
        criterion = nn.CrossEntropyLoss()
        
        # Get current loss
        inputs, targets = next(iter(dummy_dataloader))
        with torch.no_grad():
            output = simple_model(inputs)
            expected_loss = criterion(output, targets).item()
        
        # Trace landscape
        alphas, losses = trace_1d_loss(
            simple_model, dummy_dataloader, criterion, device, steps=21
        )
        
        # Find loss at alpha=0 (middle point)
        middle_idx = len(alphas) // 2
        actual_loss = losses[middle_idx]
        
        assert abs(actual_loss - expected_loss) < 1e-4
