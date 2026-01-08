__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your-email@example.com"

# Core API - Main user-facing class
from .core import QuantDiagnostic

# Quantization primitives
from .quantization import (
    FakeQuantOp,
    QuantLinear,
    replace_linear_layers
)

# Geometry utilities
from .geometry import (
    get_random_direction,
    trace_1d_loss
)

# Hessian analysis
from .hessian import power_iteration

# Plotting utilities
from .plotting import plot_overlay

from .core import QuantDiagnostic
from .quantization import FakeQuantOp, QuantLinear, replace_linear_layers

# Define public API
__all__ = [
    # Main API
    'QuantDiagnostic',
    
    # Quantization
    'FakeQuantOp',
    'QuantLinear',
    'replace_linear_layers',
    
    # Geometry
    'get_random_direction',
    'trace_1d_loss',
    
    # Hessian
    'power_iteration',
    
    # Plotting
    'plot_overlay',
]

# Package metadata
__description__ = "Zero-dependency quantization diagnostic toolkit for neural networks"
__url__ = "https://github.com/s-sairam/quant-lens"
__license__ = "MIT"

# Version info
def get_version():
    """Returns the current version of quant-lens."""
    return __version__

# Quick start banner
def print_info():
    """Prints package information."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     quant-lens v{__version__}                      ║
║          Quantization Diagnostic Toolkit                     ║
╚══════════════════════════════════════════════════════════════╝

Quick Start:
    from quant_lens import QuantDiagnostic
    
    diagnostic = QuantDiagnostic(model, dataloader)
    diagnostic.add_int8_model()
    metrics = diagnostic.run_analysis()
    diagnostic.plot()

Documentation: {__url__}
    """)
