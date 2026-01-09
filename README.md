# 🔬 quant-lens

**A Zero-Dependency Quantization Diagnostic Toolkit**

Visualize loss landscapes and Hessian sharpness to diagnose "bit collapse" in quantized neural networks.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://github.com/S-Sairam/quant-lens/actions/workflows/tests.yml/badge.svg)

---

## 🎯 Why quant-lens?

When you quantize a model from FP32 → INT8, you're compressing billions of real numbers into discrete buckets. This **"bit collapse"** can:

- ✅ **Sharpen** loss minima (making them fragile)
- ✅ **Distort** the loss landscape geometry
- ✅ **Destroy** generalization without obvious warning signs

**quant-lens** lets you *see* these effects before they ruin your deployment.

---

## 🚀 Quick Start

### Installation

```bash
pip install quant-lens
```

### Basic Usage

```python
from quant_lens import QuantDiagnostic
import torch
from torch.utils.data import DataLoader

# 1. Load your FP32 model
model = torch.load('my_model.pth')

# 2. Prepare calibration data
dataloader = DataLoader(dataset, batch_size=64)

# 3. Initialize diagnostic
diagnostic = QuantDiagnostic(model, dataloader, device='cuda')

# 4. Add quantized variant (auto-generated)
diagnostic.add_int8_model()

# 5. Run analysis
metrics = diagnostic.run_analysis()

# 6. Generate visualization
diagnostic.plot(save_path='bit_collapse.png')
```

**Output:**
```
╔═══════════════════════════════════════════╗
║    QUANTIZATION DIAGNOSTICS RESULTS       ║
╚═══════════════════════════════════════════╝

📊 SHARPNESS COMPARISON:
   FP32         λ_max = 0.234567
   Int8         λ_max = 0.876543

   Sharpness Ratio (Int8/FP32): 3.74x
   ⚠️  Quantization significantly increased sharpness!
```

---

## 📊 What You Get

### 1. **Hessian Spectrum Analysis**
Computes the top eigenvalue (λ_max) of the Hessian using Power Iteration:

- **Low λ_max** → Flat minimum (good generalization)
- **High λ_max** → Sharp minimum (poor generalization)
- **Ratio > 1.5x** → Quantization degraded the optimum

### 2. **Loss Landscape Visualization**
Traces the loss surface along a random direction using **Filter Normalization** (Li et al., 2018):

<img width="4764" height="1764" alt="bit_collapse_analysis" src="https://github.com/user-attachments/assets/fb2d344f-1bfc-44fa-8d2a-563141ba7dc0" />


**Blue Line (FP32):** Wide, smooth valley ✅  
**Red Line (Int8):** Narrow, jagged ravine ⚠️

---

## 🧠 How It Works

### Architecture Overview

```
quant-lens/
├── quantization.py   # FakeQuant with STE (Straight-Through Estimator)
├── geometry.py       # Loss landscape tracing with filter normalization
├── hessian.py        # Power iteration for Hessian eigenvalues
├── plotting.py       # Matplotlib visualization
└── core.py           # Main API (QuantDiagnostic class)
```

### Key Innovations

#### 1. **Straight-Through Estimator (STE)**
Quantization is non-differentiable. We use STE to approximate gradients:

```python
# Forward: Quantize
x_quant = round(x / scale) * scale

# Backward: Pretend it's identity
grad_x = grad_output  # Pass through unchanged
```

#### 2. **Filter Normalization**
Neural networks with BatchNorm are scale-invariant. We normalize random directions to ensure fair comparisons:

```python
d_ij = (d_ij / ||d_ij||) * ||θ_ij||
```

This removes the artificial "smoothing" effect caused by large weights.

#### 3. **Power Iteration**
Computing the full Hessian is prohibitively expensive. We use power iteration to find the dominant eigenvalue:

```python
for _ in range(20):
    v = H @ v  # Hessian-vector product
    v = v / ||v||
λ_max = v^T @ H @ v  # Rayleigh quotient
```

---

## 🔬 Advanced Usage

### Multiple Quantization Schemes

```python
diagnostic = QuantDiagnostic(model_fp32, dataloader)

# Test different bit widths
diagnostic.add_int8_model(num_bits=8, name="Int8")
diagnostic.add_int8_model(num_bits=4, name="Int4")
diagnostic.add_int8_model(num_bits=2, name="Int2")

metrics = diagnostic.run_analysis()
diagnostic.plot(save_path='multibit_analysis.png')
```

### Custom Quantized Models

```python
# If you already have a quantized model
model_custom = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

diagnostic.add_int8_model(model_int8=model_custom, name="CustomInt8")
```

### Fine-Tuning the Analysis

```python
metrics = diagnostic.run_analysis(
    landscape_steps=50,    # More points = smoother curve
    hessian_iters=30       # More iterations = better eigenvalue
)
```

---

## 📖 Theoretical Background

### Filter Normalization (Li et al., 2018)
> "Visualizing the Loss Landscape of Neural Nets" - NeurIPS 2018

Key insight: Networks with BatchNorm exhibit scale invariance. Multiplying weights by a constant doesn't change the function. Filter normalization removes this artifact.

### Straight-Through Estimator (Bengio et al., 2013)
> "Estimating or Propagating Gradients Through Stochastic Neurons"

The STE allows training discrete networks by approximating ∂f/∂x ≈ 1 during backpropagation, even though the true gradient is zero almost everywhere.

### Power Iteration (Golub & Van Loan, 2013)
Classic algorithm for finding dominant eigenvectors. Converges geometrically at rate |λ₂/λ₁|.

---

## ⚙️ Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy ≥ 1.20
- Matplotlib ≥ 3.5

**Note:** No external dependencies like `loss-landscapes` or `PyHessian`. Everything is self-contained to avoid version conflicts and OOM errors.

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
```python
# Use smaller batches for calibration
dataloader = DataLoader(dataset, batch_size=16)  # Reduce from 64
```

### Issue: Power Iteration Fails
```python
# Increase iterations or check for numerical instability
metrics = diagnostic.run_analysis(hessian_iters=50)
```

### Issue: Flat Landscape (No Variation)
```python
# Increase distance to explore wider region
diagnostic.add_int8_model()
x, y = trace_1d_loss(model, loader, criterion, device, distance=1.0)
```

---

## 🎓 Citation

If you use quant-lens in your research, please cite:

```bibtex
@software{quant_lens_2026,
  title={quant-lens: Diagnostic Toolkit for Neural Network Quantization},
  author={Sairam S},
  year={2026},
  url={https://github.com/s-sairam/quant-lens}
}
```

**References:**
- Li et al. (2018) - Visualizing the Loss Landscape of Neural Nets
- Bengio et al. (2013) - Estimating Gradients Through Stochastic Neurons
- Yin et al. (2019) - Understanding Straight-Through Estimators

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

**Roadmap:**
- [ ] 2D landscape visualization
- [ ] Batch Normalization statistics tracking
- [ ] Layer-wise sharpness analysis
- [ ] Integration with torchao/QAT
- [ ] Support for activation quantization

---

## 💬 Contact

Questions? Open an issue or reach out at: saisr2206@gmail.com

**Happy quantizing! 🚀**

