"""
QUICK TEST SCRIPT - RUN THIS FIRST
=============================================================================
This script performs a quick sanity check of all quant-lens components
without requiring pytest. Run this before running the full test suite.

Usage: python quick_test.py

If this passes, your installation is correct!
=============================================================================
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def print_header(text):
    """Pretty print section headers"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_check(text, passed):
    """Print check result"""
    symbol = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    print(f"  [{symbol}] {text}: {status}")
    return passed


def test_imports():
    """Test 1: Can we import all components?"""
    print_header("TEST 1: Imports")
    
    all_passed = True
    
    try:
        from quant_lens import QuantDiagnostic
        all_passed &= print_check("Import QuantDiagnostic", True)
    except ImportError as e:
        all_passed &= print_check(f"Import QuantDiagnostic: {e}", False)
    
    try:
        from quant_lens import FakeQuantOp, QuantLinear
        all_passed &= print_check("Import Quantization components", True)
    except ImportError as e:
        all_passed &= print_check(f"Import Quantization: {e}", False)
    
    try:
        from quant_lens import trace_1d_loss, power_iteration
        all_passed &= print_check("Import Analysis functions", True)
    except ImportError as e:
        all_passed &= print_check(f"Import Analysis: {e}", False)
    
    return all_passed


def test_quantization():
    """Test 2: Does quantization work?"""
    print_header("TEST 2: Quantization (FakeQuantOp)")
    
    from quant_lens.quantization import FakeQuantOp
    
    all_passed = True
    
    # Test 2.1: Forward pass
    try:
        x = torch.tensor([[1.5, 2.7, 3.9]])
        quantized = FakeQuantOp.apply(x)
        all_passed &= print_check("Forward pass completes", True)
    except Exception as e:
        all_passed &= print_check(f"Forward pass: {e}", False)
        return False
    
    # Test 2.2: Quantization changes values
    changed = not torch.allclose(x, quantized, atol=1e-6)
    all_passed &= print_check(f"Quantization modifies values {changed} ", True)
    
    # Test 2.3: Gradient flows (STE)
    try:
        x = torch.tensor([[1.5, 2.7]], requires_grad=True)
        y = FakeQuantOp.apply(x)
        y.sum().backward()
        has_grad = x.grad is not None and not torch.allclose(x.grad, torch.zeros_like(x.grad))
        all_passed &= print_check("Gradients flow through (STE)", has_grad)
    except Exception as e:
        all_passed &= print_check(f"Gradient flow: {e}", False)
    
    # Test 2.4: Deterministic
    x = torch.tensor([[1.5, 2.5]])
    result1 = FakeQuantOp.apply(x)
    result2 = FakeQuantOp.apply(x)
    deterministic = torch.allclose(result1, result2)
    all_passed &= print_check("Deterministic output", deterministic)
    
    return all_passed


def test_model_replacement():
    """Test 3: Does model replacement work?"""
    print_header("TEST 3: Model Replacement")
    
    from quant_lens.quantization import replace_linear_layers, QuantLinear
    
    all_passed = True
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    try:
        quant_model = replace_linear_layers(model, num_bits=8)
        all_passed &= print_check("Model replacement completes", True)
    except Exception as e:
        all_passed &= print_check(f"Model replacement: {e}", False)
        return False
    
    # Check QuantLinear layers exist
    quant_count = sum(1 for m in quant_model.modules() if isinstance(m, QuantLinear))
    expected_count = 2  # Two Linear layers in original model
    correct_count = (quant_count == expected_count)
    all_passed &= print_check(f"Converted {quant_count} layers (expected {expected_count})", correct_count)
    
    # Check weights preserved
    original_weight = model[0].weight.data
    replaced_weight = quant_model[0].weight.data
    preserved = torch.allclose(original_weight, replaced_weight)
    all_passed &= print_check("Weights preserved after conversion", preserved)
    
    return all_passed


def test_loss_landscape():
    """Test 4: Does loss landscape tracing work?"""
    print_header("TEST 4: Loss Landscape Tracing")
    
    from quant_lens.geometry import trace_1d_loss
    all_passed = True
    
    # Setup
    model = nn.Sequential(nn.Linear(10, 5))

    import torch as t
    X = t.randn(32, 10)

    y = t.randint(0, 5, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    criterion = nn.CrossEntropyLoss()
    
    try:
        alphas, losses = trace_1d_loss(model, loader, criterion, 'cpu', steps=11)
        all_passed &= print_check("Landscape tracing completes", True)
    except Exception as e:
        all_passed &= print_check(f"Landscape tracing: {e}", False)
        return False
    
    # Check output format
    correct_length = (len(alphas) == 11 and len(losses) == 11)
    all_passed &= print_check(f"Returns {len(losses)} points (expected 11)", correct_length)
    
    # Check alpha range
    alpha_range_ok = (abs(alphas[0] + 0.5) < 0.01 and abs(alphas[-1] - 0.5) < 0.01)
    all_passed &= print_check(f"Alpha range [{alphas[0]:.2f}, {alphas[-1]:.2f}]", alpha_range_ok)
    
    # Check weights restored
    original_weight = model[0].weight.data.clone()
    trace_1d_loss(model, loader, criterion, 'cpu', steps=5)
    current_weight = model[0].weight.data
    restored = torch.allclose(original_weight, current_weight)
    all_passed &= print_check("Original weights restored", restored)
    
    return all_passed


def test_hessian():
    """Test 5: Does Hessian computation work?"""
    print_header("TEST 5: Hessian Eigenvalue (Sharpness)")
    
    import torch 
    from quant_lens.hessian import power_iteration
    
    all_passed = True
    
    # Setup
    model = nn.Sequential(nn.Linear(10, 5))
    X = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    criterion = nn.CrossEntropyLoss()
    
    try:
        eigenvalue = power_iteration(model, loader, criterion, 'cpu', num_iters=5)
        all_passed &= print_check("Hessian computation completes", True)
    except Exception as e:
        all_passed &= print_check(f"Hessian computation: {e}", False)
        return False
    
    # Check eigenvalue is valid
    is_number = isinstance(eigenvalue, (int, float))
    all_passed &= print_check(f"Returns number (λ_max = {eigenvalue:.6f})", is_number)
    
    # Check non-negative
    non_negative = (eigenvalue >= 0)
    all_passed &= print_check("Eigenvalue is non-negative", non_negative)
    
    # Check reproducibility
    torch.manual_seed(42)
    eigen1 = power_iteration(model, loader, criterion, 'cpu', num_iters=10)
    torch.manual_seed(42)
    eigen2 = power_iteration(model, loader, criterion, 'cpu', num_iters=10)
    reproducible = abs(eigen1 - eigen2) < 1e-6
    all_passed &= print_check("Reproducible with seed", reproducible)
    
    return all_passed


def test_core_api():
    """Test 6: Does the main API work?"""
    print_header("TEST 6: Core API (QuantDiagnostic)")
    
    from quant_lens import QuantDiagnostic
    from quant_lens.quantization import QuantLinear
    import torch as t

    all_passed = True
    
    # Setup
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    X = t.randn(64, 10)
    y = t.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)
    
    # Test initialization
    try:
        diagnostic = QuantDiagnostic(model, loader, device='cpu')
        all_passed &= print_check("API initialization", True)
    except Exception as e:
        all_passed &= print_check(f"API initialization: {e}", False)
        return False
    
    # Test add_int8_model
    try:
        diagnostic.add_int8_model()
        has_int8 = 'Int8' in diagnostic.models
        all_passed &= print_check("Add Int8 model", has_int8)
    except Exception as e:
        all_passed &= print_check(f"Add Int8 model: {e}", False)
        return False
    
    # Check Int8 model has QuantLinear
    has_quant = any(isinstance(m, QuantLinear) 
                    for m in diagnostic.models['Int8'].modules())
    all_passed &= print_check("Int8 model is quantized", has_quant)
    
    # Test run_analysis
    try:
        metrics = diagnostic.run_analysis(landscape_steps=5, hessian_iters=3)
        all_passed &= print_check("Run analysis completes", True)
    except Exception as e:
        all_passed &= print_check(f"Run analysis: {e}", False)
        return False
    
    # Check metrics structure
    has_fp32 = 'FP32' in metrics
    has_int8 = 'Int8' in metrics
    has_sharpness = 'sharpness' in metrics.get('FP32', {})
    all_passed &= print_check("Metrics structure correct", has_fp32 and has_int8 and has_sharpness)
    
    if has_fp32 and has_int8 and has_sharpness:
        fp32_sharp = metrics['FP32']['sharpness']
        int8_sharp = metrics['Int8']['sharpness']
        ratio = int8_sharp / fp32_sharp if fp32_sharp > 0 else 0
        print(f"      FP32 sharpness: {fp32_sharp:.6f}")
        print(f"      Int8 sharpness: {int8_sharp:.6f}")
        print(f"      Ratio: {ratio:.2f}x")
    
    return all_passed


def test_end_to_end():
    """Test 7: Full end-to-end workflow"""
    print_header("TEST 7: End-to-End Workflow")
    
    from quant_lens import QuantDiagnostic
    import tempfile
    import os
    import torch 

    all_passed = True
    
    # Setup
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    X = torch.randn(64, 10)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)
    
    try:
        # Full workflow
        diagnostic = QuantDiagnostic(model, loader, device='cpu')
        diagnostic.add_int8_model()
        metrics = diagnostic.run_analysis(landscape_steps=5, hessian_iters=3)
        
        # Save plot to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_plot.png')
            diagnostic.plot(save_path=save_path)
            plot_exists = os.path.exists(save_path)
            all_passed &= print_check("Complete workflow with plot", plot_exists)
    except Exception as e:
        all_passed &= print_check(f"End-to-end workflow: {e}", False)
        return False
    
    return all_passed


def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                   QUANT-LENS QUICK TEST                          ║
║              Sanity Check for All Components                     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Quantization", test_quantization()))
    results.append(("Model Replacement", test_model_replacement()))
    results.append(("Loss Landscape", test_loss_landscape()))
    results.append(("Hessian", test_hessian()))
    results.append(("Core API", test_core_api()))
    results.append(("End-to-End", test_end_to_end()))
    
    # Print summary
    print_header("SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:25s} {status}")
    
    print("\n" + "="*70)
    if passed == total:
        print(f"  ✅ ALL {total} TESTS PASSED!")
        print("  Your quant-lens installation is working correctly.")
        print("\n  Next steps:")
        print("  1. Run full test suite: pytest tests/ -v")
        print("  2. Try examples: python examples/demo_simple.py")
        print("="*70)
        return 0
    else:
        print(f"  ⚠️  {passed}/{total} TESTS PASSED, {total - passed} FAILED")
        print("  Please check the failed tests above.")
        print("\n  Troubleshooting:")
        print("  - Ensure quant-lens is installed: pip install -e .")
        print("  - Check dependencies: pip install torch numpy matplotlib")
        print("  - Review error messages for specific issues")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
