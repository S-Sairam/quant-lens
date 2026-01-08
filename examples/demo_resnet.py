"""
Complete Working Example: Using quant-lens with ResNet on CIFAR-10

This example demonstrates:
1. Loading a pretrained model
2. Setting up a calibration dataset
3. Running the diagnostic analysis
4. Interpreting the results

Run: python examples/demo_resnet.py
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Assuming quant_lens is installed or in PYTHONPATH
from quant_lens.core import QuantDiagnostic


# ============================================================================
# Simple ResNet-18 Model Definition
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ============================================================================
# Data Loading
# ============================================================================

def load_cifar10_calibration(batch_size=64, num_samples=512):
    """
    Loads a small subset of CIFAR-10 for calibration.
    
    Args:
        batch_size: Batch size for DataLoader
        num_samples: Number of samples to use (for speed)
    
    Returns:
        DataLoader with calibration data
    """
    print(f"\n📦 Loading CIFAR-10 calibration data ({num_samples} samples)...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Download CIFAR-10 (only downloads once)
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Use subset for speed
    indices = torch.randperm(len(dataset))[:num_samples]
    subset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    print(f"✓ Loaded {len(subset)} samples in {len(dataloader)} batches")
    return dataloader


# ============================================================================
# Model Training Helper (Optional - for getting a pretrained model)
# ============================================================================

def train_resnet_cifar10(epochs=5, device='cuda'):
    """
    Trains a ResNet-18 on CIFAR-10 for a few epochs.
    This is optional - you can skip if you have a pretrained model.
    """
    print("\n🏋️ Training ResNet-18 on CIFAR-10...")
    
    # Data
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    # Model
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'  Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss/100:.3f}')
                running_loss = 0.0
    
    print("✓ Training complete!")
    
    # Save model
    torch.save(model.state_dict(), 'resnet18_cifar10.pth')
    print("✓ Model saved to: resnet18_cifar10.pth")
    
    return model


# ============================================================================
# Main Diagnostic Workflow
# ============================================================================

def run_quantization_diagnostic():
    """
    Main function: Runs the complete quant-lens diagnostic workflow.
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  QUANT-LENS DIAGNOSTIC DEMO                      ║
║           Analyzing Quantization Effects on ResNet-18            ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    
    # Step 1: Load or train model
    model_fp32 = ResNet18()
    
    try:
        # Try to load pretrained model
        model_fp32.load_state_dict(torch.load('resnet18_cifar10.pth', 
                                              map_location=device))
        print("✓ Loaded pretrained model from resnet18_cifar10.pth")
    except FileNotFoundError:
        print("⚠️  No pretrained model found. Training from scratch...")
        print("   (This will take ~5-10 minutes on GPU)")
        model_fp32 = train_resnet_cifar10(epochs=5, device=device)
    
    model_fp32.eval()
    
    # Step 2: Load calibration data
    dataloader = load_cifar10_calibration(batch_size=64, num_samples=256)
    
    # Step 3: Initialize diagnostic
    print("\n🔬 Initializing quant-lens diagnostic...")
    diagnostic = QuantDiagnostic(model_fp32, dataloader, device=device)
    
    # Step 4: Add quantized variants
    print("\n📊 Adding quantized model variants...")
    diagnostic.add_int8_model(num_bits=8, name="Int8")
    diagnostic.add_int8_model(num_bits=4, name="Int4")  # More aggressive
    
    # Step 5: Run analysis
    metrics = diagnostic.run_analysis(
        landscape_steps=30,  # More points for smoother curve
        hessian_iters=20     # Standard for convergence
    )
    
    # Step 6: Generate visualization
    print("\n🎨 Generating visualization...")
    diagnostic.plot(save_path='resnet18_quantization_analysis.png', dpi=300)
    
    # Step 7: Detailed interpretation
    print("\n" + "="*70)
    print("📋 DETAILED INTERPRETATION")
    print("="*70)
    
    fp32_sharp = metrics['FP32']['sharpness']
    int8_sharp = metrics['Int8']['sharpness']
    int4_sharp = metrics['Int4']['sharpness']
    
    print(f"\n1. Sharpness Analysis (λ_max - Top Hessian Eigenvalue):")
    print(f"   FP32: {fp32_sharp:.6f}")
    print(f"   Int8: {int8_sharp:.6f} ({int8_sharp/fp32_sharp:.2f}x)")
    print(f"   Int4: {int4_sharp:.6f} ({int4_sharp/fp32_sharp:.2f}x)")
    
    if int8_sharp / fp32_sharp > 2.0:
        print(f"\n   ⚠️  WARNING: Int8 quantization increased sharpness by {int8_sharp/fp32_sharp:.1f}x!")
        print(f"   → This suggests the model may be more sensitive to perturbations")
        print(f"   → Consider: Quantization-Aware Training (QAT) to flatten the minimum")
    else:
        print(f"\n   ✓ Int8 quantization preserved flatness reasonably well")
    
    if int4_sharp / fp32_sharp > 5.0:
        print(f"\n   🚨 CRITICAL: Int4 quantization severely sharpened the minimum!")
        print(f"   → 4-bit quantization may be too aggressive for this model")
        print(f"   → Recommendations:")
        print(f"     • Use mixed precision (4-bit for some layers, 8-bit for critical ones)")
        print(f"     • Apply QAT with longer fine-tuning")
        print(f"     • Consider per-channel quantization instead of per-tensor")
    
    print(f"\n2. Loss Landscape Geometry:")
    print(f"   Check the generated plot: resnet18_quantization_analysis.png")
    print(f"   Look for:")
    print(f"   • Wide valleys → Robust to weight perturbations (good)")
    print(f"   • Narrow ravines → Fragile to noise (bad)")
    print(f"   • Asymmetry → Quantization distorted the geometry")
    
    print("\n" + "="*70)
    print("✅ Analysis complete! Check the output files:")
    print("   • resnet18_quantization_analysis.png - Visual comparison")
    print("   • metrics dictionary - Numerical results")
    print("="*70)
    
    return metrics


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run the diagnostic
    metrics = run_quantization_diagnostic()
    
    print("\n🎉 Demo complete!")
    print("\nNext steps:")
    print("1. Examine the generated plot")
    print("2. If sharpness ratio > 2x, consider Quantization-Aware Training")
    print("3. Try different quantization schemes (per-channel, mixed precision)")
    print("4. Use quant-lens to compare before/after QAT")
