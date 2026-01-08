import pytest
import os
import matplotlib.pyplot as plt
from quant_lens.plotting import plot_overlay


class TestPlotOverlay:
    """Tests for visualization"""
    
    def test_creates_plot_file(self, tmp_path):
        """Should create a PNG file"""
        traces = {
            'Model1': ([-1, 0, 1], [2.0, 1.0, 2.0]),
            'Model2': ([-1, 0, 1], [2.5, 1.5, 2.5])
        }
        
        save_path = tmp_path / "test_plot.png"
        plot_overlay(traces, str(save_path), dpi=50)
        
        assert save_path.exists()
    
    def test_handles_multiple_models(self, tmp_path):
        """Should handle multiple model traces"""
        traces = {
            'FP32': ([-1, 0, 1], [2.0, 1.0, 2.0]),
            'Int8': ([-1, 0, 1], [2.5, 1.5, 2.5]),
            'Int4': ([-1, 0, 1], [3.0, 2.0, 3.0])
        }
        
        save_path = tmp_path / "multi_model.png"
        
        try:
            plot_overlay(traces, str(save_path))
            assert True
        except Exception as e:
            pytest.fail(f"plot_overlay failed with multiple models: {e}")
    
    def test_closes_figures(self, tmp_path):
        """Should close matplotlib figures to avoid memory leaks"""
        traces = {'Model': ([0, 1], [1.0, 1.5])}
        
        initial_figs = len(plt.get_fignums())
        plot_overlay(traces, str(tmp_path / "test.png"))
        final_figs = len(plt.get_fignums())
        
        # Should not leave figures open
        assert final_figs == initial_figs
