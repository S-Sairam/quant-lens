#!/usr/bin/env python3
"""
Full-Scale Demonstration: Bit Collapse Analysis with quant-lens
================================================================

This script demonstrates the complete workflow of analyzing quantization
effects on ResNet-18 using the quant-lens library.

Compares three scenarios:
1. FP32 Baseline (SGD)
2. Int8 Quantized (SGD) - Shows bit collapse
3. Int8 Quantized (SAM) - Shows sharpness-aware training fixes collapse

Run: python examples/full_scale_demo.py

Based on: run.py (overnight mode)
Updated: Uses quant-lens library for all diagnostics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import random
import copy

# Import our quant-lens library
from quant_lens import QuantDiagnostic
from quant_lens.quantization import replace_linear_layers


# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    """Experiment configuration"""
    # System
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    SAVE_DIR = "quant_lens_results"
    
    # Training
    TRAIN_BATCH_SIZE = 64
    EPOCHS = 15              # Deep fine-tuning
    LR = 0.001
    MOMENTUM = 0.9
    
    # SAM Settings
    SAM_RHO = 0.05
    
    # quant-lens Analysis Settings
    ANALYSIS_BATCH_SIZE = 32
    ANALYSIS_SAMPLES = 256   # Subset for faster analysis
    LANDSCAPE_STEPS = 50
    HESSIAN_ITERS = 30       # High precision
    
    # Visualization
    PLOT_DISTANCE = 1.0


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.DEVICE == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# ==========================================
# 2. SAM OPTIMIZER
# ==========================================
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer"""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["e_w"] = p.grad * scale
                p.add_(self.state[p]["e_w"])
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        stack = [
            p.grad.norm(p=2).to(p.device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(stack), p=2)


# ==========================================
# 3. DATA LOADING
# ==========================================
def get_loaders():
    """Create CIFAR-10 dataloaders"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Training set
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Analysis set (smaller subset for faster analysis)
    from torch.utils.data import Subset
    indices = torch.randperm(len(trainset))[:Config.ANALYSIS_SAMPLES]
    analysis_set = Subset(trainset, indices)
    
    analysis_loader = torch.utils.data.DataLoader(
        analysis_set,
        batch_size=Config.ANALYSIS_BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, analysis_loader


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_loop(model, loader, optimizer, use_sam=False):
    """Single epoch training"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if use_sam:
            optimizer.first_step(zero_grad=True)
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


# ==========================================
# 5. ANALYSIS USING QUANT-LENS
# ==========================================
def analyze_model(model, analysis_loader, name):
    """
    Analyze a model using quant-lens library.
    
    Returns:
        dict: Contains sharpness, landscape data, etc.
    """
    print(f"\n[quant-lens] Analyzing {name}...")
    
    # Create diagnostic
    diagnostic = QuantDiagnostic(
        model,
        analysis_loader,
        device=Config.DEVICE
    )
    
    # Run analysis
    metrics = diagnostic.run_analysis(
        landscape_steps=Config.LANDSCAPE_STEPS,
        hessian_iters=Config.HESSIAN_ITERS
    )
    
    # Extract results
    sharpness = metrics['FP32']['sharpness']
    alphas, losses = diagnostic.traces['FP32']
    
    print(f"   ✓ Sharpness (λ_max): {sharpness:.4f}")
    
    return {
        'sharpness': sharpness,
        'alphas': alphas,
        'losses': losses,
        'metrics': metrics
    }


# ==========================================
# 6. MAIN EXPERIMENT
# ==========================================
def run_experiment():
    """Main experiment workflow"""
    set_seed(Config.SEED)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    print("="*70)
    print("FULL-SCALE BIT COLLAPSE ANALYSIS WITH QUANT-LENS")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Save directory: {Config.SAVE_DIR}")
    print()
    
    # Get data
    train_loader, analysis_loader = get_loaders()
    print(f"✓ Loaded CIFAR-10")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Analysis samples: {len(analysis_loader.dataset)}")
    
    # Dictionary to store all results
    results = {}
    
    # =====================================================
    # PHASE 1: FP32 BASELINE
    # =====================================================
    print("\n" + "="*70)
    print("PHASE 1: FP32 BASELINE (SGD)")
    print("="*70)
    
    model_fp32 = resnet18(weights=ResNet18_Weights.DEFAULT)
    model_fp32.fc = nn.Linear(512, 10)
    model_fp32 = model_fp32.to(Config.DEVICE)
    
    opt_fp32 = optim.SGD(
        model_fp32.parameters(),
        lr=Config.LR,
        momentum=Config.MOMENTUM
    )
    
    print(f"Training FP32 model for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        loss = train_loop(model_fp32, train_loader, opt_fp32, use_sam=False)
        print(f"  Epoch {epoch+1}/{Config.EPOCHS}: Loss = {loss:.4f}")
    
    # Analyze with quant-lens
    results['fp32'] = analyze_model(model_fp32, analysis_loader, "FP32 Baseline")
    
    # Save converged state
    converged_state = copy.deepcopy(model_fp32.state_dict())
    
    # Cleanup
    del model_fp32, opt_fp32
    torch.cuda.empty_cache()
    gc.collect()
    
    # =====================================================
    # PHASE 2: INT8 WITH SGD (Bit Collapse)
    # =====================================================
    print("\n" + "="*70)
    print("PHASE 2: INT8 QUANTIZED (SGD) - Expect Bit Collapse")
    print("="*70)
    
    model_int8_sgd = resnet18(weights=None)
    model_int8_sgd.fc = nn.Linear(512, 10)
    model_int8_sgd.load_state_dict(converged_state)  # Start from FP32 weights
    
    # Apply quantization using quant-lens
    print("Applying quantization (8-bit)...")
    model_int8_sgd = replace_linear_layers(model_int8_sgd, num_bits=8)
    model_int8_sgd = model_int8_sgd.to(Config.DEVICE)
    
    opt_int8_sgd = optim.SGD(
        model_int8_sgd.parameters(),
        lr=Config.LR,
        momentum=Config.MOMENTUM
    )
    
    print(f"Training Int8 model (SGD) for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        loss = train_loop(model_int8_sgd, train_loader, opt_int8_sgd, use_sam=False)
        print(f"  Epoch {epoch+1}/{Config.EPOCHS}: Loss = {loss:.4f}")
    
    # Analyze with quant-lens
    results['int8_sgd'] = analyze_model(model_int8_sgd, analysis_loader, "Int8 SGD")
    
    # Cleanup
    del model_int8_sgd, opt_int8_sgd
    torch.cuda.empty_cache()
    gc.collect()
    
    # =====================================================
    # PHASE 3: INT8 WITH SAM (Recovery)
    # =====================================================
    print("\n" + "="*70)
    print("PHASE 3: INT8 QUANTIZED (SAM) - Sharpness-Aware Training")
    print("="*70)
    
    model_int8_sam = resnet18(weights=None)
    model_int8_sam.fc = nn.Linear(512, 10)
    model_int8_sam.load_state_dict(converged_state)  # Start from FP32 weights
    
    # Apply quantization
    print("Applying quantization (8-bit)...")
    model_int8_sam = replace_linear_layers(model_int8_sam, num_bits=8)
    model_int8_sam = model_int8_sam.to(Config.DEVICE)
    
    opt_int8_sam = SAM(
        model_int8_sam.parameters(),
        optim.SGD,
        rho=Config.SAM_RHO,
        lr=Config.LR,
        momentum=Config.MOMENTUM
    )
    
    print(f"Training Int8 model (SAM) for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        loss = train_loop(model_int8_sam, train_loader, opt_int8_sam, use_sam=True)
        print(f"  Epoch {epoch+1}/{Config.EPOCHS}: Loss = {loss:.4f}")
    
    # Analyze with quant-lens
    results['int8_sam'] = analyze_model(model_int8_sam, analysis_loader, "Int8 SAM")
    
    # Cleanup
    del model_int8_sam, opt_int8_sam
    torch.cuda.empty_cache()
    gc.collect()
    
    # =====================================================
    # RESULTS & VISUALIZATION
    # =====================================================
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print("\nSharpness Comparison (λ_max):")
    print(f"  FP32 Baseline:  {results['fp32']['sharpness']:.4f}")
    print(f"  Int8 SGD:       {results['int8_sgd']['sharpness']:.4f}")
    print(f"  Int8 SAM:       {results['int8_sam']['sharpness']:.4f}")
    
    # Calculate ratios
    sgd_ratio = results['int8_sgd']['sharpness'] / results['fp32']['sharpness']
    sam_ratio = results['int8_sam']['sharpness'] / results['fp32']['sharpness']
    
    print(f"\nSharpness Ratios:")
    print(f"  Int8 SGD / FP32: {sgd_ratio:.2f}x")
    print(f"  Int8 SAM / FP32: {sam_ratio:.2f}x")
    
    if sgd_ratio > 1.5:
        print(f"\n⚠️  WARNING: Int8 SGD shows significant sharpening ({sgd_ratio:.2f}x)!")
        print("   This indicates 'bit collapse' - the quantized model found a sharp minimum.")
    
    if sam_ratio < sgd_ratio * 0.8:
        print(f"\n✅ SUCCESS: SAM reduced sharpness by {(1 - sam_ratio/sgd_ratio)*100:.1f}%!")
        print("   Sharpness-aware training successfully flattened the minimum.")
    
    # Save metrics
    with open(f"{Config.SAVE_DIR}/metrics.txt", "w") as f:
        f.write("Bit Collapse Analysis Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"FP32 Baseline Sharpness:  {results['fp32']['sharpness']:.6f}\n")
        f.write(f"Int8 SGD Sharpness:       {results['int8_sgd']['sharpness']:.6f}\n")
        f.write(f"Int8 SAM Sharpness:       {results['int8_sam']['sharpness']:.6f}\n\n")
        f.write(f"SGD Ratio (Int8/FP32):    {sgd_ratio:.4f}x\n")
        f.write(f"SAM Ratio (Int8/FP32):    {sam_ratio:.4f}x\n")
    
    print(f"\n✓ Metrics saved to {Config.SAVE_DIR}/metrics.txt")
    
    # =====================================================
    # GENERATE VISUALIZATION
    # =====================================================
    print("\nGenerating loss landscape visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss Landscapes
    ax1.plot(
        results['fp32']['alphas'],
        results['fp32']['losses'],
        label='FP32 (SGD)',
        color='#2E86AB',
        linewidth=2.5,
        linestyle='--'
    )
    ax1.plot(
        results['int8_sgd']['alphas'],
        results['int8_sgd']['losses'],
        label='Int8 (SGD) - Bit Collapse',
        color='#A23B72',
        linewidth=2.5
    )
    ax1.plot(
        results['int8_sam']['alphas'],
        results['int8_sam']['losses'],
        label='Int8 (SAM) - Recovery',
        color='#06A77D',
        linewidth=2.5
    )
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('α (Interpolation Coefficient)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Landscape: Bit Collapse & Recovery', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Zoom into basin
    y_min = min(
        min(results['fp32']['losses']),
        min(results['int8_sgd']['losses']),
        min(results['int8_sam']['losses'])
    )
    y_max = y_min + 5.0
    ax1.set_ylim(y_min - 0.5, y_max)
    
    # Plot 2: Sharpness Comparison
    methods = ['FP32\nBaseline', 'Int8\nSGD', 'Int8\nSAM']
    sharpness_values = [
        results['fp32']['sharpness'],
        results['int8_sgd']['sharpness'],
        results['int8_sam']['sharpness']
    ]
    colors = ['#2E86AB', '#A23B72', '#06A77D']
    
    bars = ax2.bar(methods, sharpness_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Sharpness (λ_max)', fontsize=12)
    ax2.set_title('Hessian Eigenvalue Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar, value in zip(bars, sharpness_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{value:.2f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.tight_layout()
    save_path = f"{Config.SAVE_DIR}/bit_collapse_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {save_path}")
    
    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {Config.SAVE_DIR}/")
    print("  - metrics.txt: Numerical results")
    print("  - bit_collapse_analysis.png: Visualization")
    print()
    print("Key Findings:")
    print(f"  • Quantization (SGD) increased sharpness by {(sgd_ratio-1)*100:.1f}%")
    print(f"  • SAM reduced quantization sharpness by {(1 - sam_ratio/sgd_ratio)*100:.1f}%")
    print()
    print("This demonstrates the 'bit collapse' phenomenon and its mitigation via SAM.")
    print("="*70)


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
