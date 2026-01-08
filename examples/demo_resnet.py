

def demo_usage():
    """
    Example demonstration of quant-lens usage.
    This would typically be in a separate file: examples/demo_resnet.py
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  QUANT-LENS DEMO SCRIPT                      ║
║          Quantization Diagnostic Tool - Example Usage        ║
╚══════════════════════════════════════════════════════════════╝

This example demonstrates how to use quant-lens to diagnose
quantization effects on a neural network model.

USAGE PATTERN:
--------------
from quant_lens.core import QuantDiagnostic

# 1. Load your FP32 model
model_fp32 = YourModel()
model_fp32.load_state_dict(torch.load('checkpoint.pth'))

# 2. Prepare a dataloader (for calibration)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. Initialize diagnostic
diagnostic = QuantDiagnostic(model_fp32, dataloader, device='cuda')

# 4. Add quantized variant (auto or manual)
diagnostic.add_int8_model()  # Auto-generates FakeQuant model

# 5. Run analysis
metrics = diagnostic.run_analysis()

# 6. Generate visualization
diagnostic.plot(save_path='my_analysis.png')

INTERPRETING RESULTS:
---------------------
• Sharpness (λ_max): Higher values indicate sharper minima
  - Sharp minima → Poor generalization
  - Flat minima → Better generalization
  - Ratio > 1.5x suggests quantization degraded the optimum

• Loss Landscape: Visual inspection of curvature
  - Steep valleys → Sensitive to perturbations
  - Wide valleys → Robust to weight changes

For more examples, see: github.com/yourrepo/quant-lens
""")


if __name__ == "__main__":
    demo_usage()

