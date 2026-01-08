import torch
import torch.nn as nn
from quant_lens.core import QuantDiagnostic
from quant_lens.quantization import QuantLinear
import pytest

class TestQuantDiagnostic:
    """Tests for main API"""
    
    def test_initialization(self, simple_model, dummy_dataloader, device):
        """Should initialize without errors"""
        diagnostic = QuantDiagnostic(simple_model, dummy_dataloader, device)
        
        assert diagnostic.model_fp32 is simple_model
        assert 'FP32' in diagnostic.models
    
    def test_add_int8_model_auto(self, simple_model, dummy_dataloader, device):
        """Should auto-generate quantized model"""
        diagnostic = QuantDiagnostic(simple_model, dummy_dataloader, device)
        diagnostic.add_int8_model()
        
        assert 'Int8' in diagnostic.models
        
        # Check it's actually quantized
        has_quant_layer = False
        for module in diagnostic.models['Int8'].modules():
            if isinstance(module, QuantLinear):
                has_quant_layer = True
                break
        
        assert has_quant_layer
    
    def test_add_int8_model_custom(self, simple_model, dummy_dataloader, device):
        """Should accept custom quantized model"""
        custom_model = nn.Sequential(nn.Linear(10, 5))
        
        diagnostic = QuantDiagnostic(simple_model, dummy_dataloader, device)
        diagnostic.add_int8_model(custom_model, name="Custom")
        
        assert 'Custom' in diagnostic.models
        assert diagnostic.models['Custom'] is custom_model
    
    def test_run_analysis_returns_metrics(self, simple_model, dummy_dataloader, device):
        """Should return metrics dictionary"""
        diagnostic = QuantDiagnostic(simple_model, dummy_dataloader, device)
        diagnostic.add_int8_model()
        
        metrics = diagnostic.run_analysis(landscape_steps=5, hessian_iters=3)
        
        assert 'FP32' in metrics
        assert 'Int8' in metrics
        assert 'sharpness' in metrics['FP32']
    
    def test_plot_requires_analysis(self, simple_model, dummy_dataloader, device, tmp_path):
        """Should fail if run_analysis not called first"""
        diagnostic = QuantDiagnostic(simple_model, dummy_dataloader, device)
        
        with pytest.raises(ValueError, match="No landscape data"):
            diagnostic.plot(save_path=str(tmp_path / "test.png"))
    
    def test_full_workflow(self, simple_model, dummy_dataloader, device, tmp_path):
        """End-to-end workflow should complete"""
        diagnostic = QuantDiagnostic(simple_model, dummy_dataloader, device)
        diagnostic.add_int8_model()
        
        # Run analysis
        metrics = diagnostic.run_analysis(landscape_steps=5, hessian_iters=3)
        
        # Generate plot
        save_path = tmp_path / "workflow_test.png"
        diagnostic.plot(save_path=str(save_path))
        
        # Verify results
        assert metrics is not None
        assert save_path.exists()
