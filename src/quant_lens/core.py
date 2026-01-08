class QuantDiagnostic:
    """
    Main diagnostic tool for quantization analysis.
    
    Usage:
        diagnostic = QuantDiagnostic(model_fp32, dataloader, device='cuda')
        diagnostic.add_int8_model()  # Auto-generates FakeQuant model
        metrics = diagnostic.run_analysis()
        diagnostic.plot(save_path='bit_collapse.png')
    """
    
    def __init__(self, model_fp32, dataloader, device='cuda'):
        """
        Args:
            model_fp32: Reference FP32 model
            dataloader: DataLoader for calibration
            device: 'cuda' or 'cpu'
        """
        self.model_fp32 = model_fp32
        self.loader = dataloader
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models = {'FP32': model_fp32}
        self.metrics = {}
        self.traces = {}
        
        print(f"[quant-lens] Initialized on device: {self.device}")
    
    def add_int8_model(self, model_int8=None, name="Int8", num_bits=8):
        """
        Adds a quantized model for comparison.
        
        Args:
            model_int8: Pre-quantized model (optional)
            name: Name for this model variant
            num_bits: Bit width for FakeQuant (default: 8)
        """
        if model_int8 is None:
            print(f"[{name}] Auto-generating FakeQuant model ({num_bits}-bit)...")
            self.models[name] = replace_linear_layers(self.model_fp32, num_bits)
        else:
            self.models[name] = model_int8
        
        print(f"[{name}] Model added to diagnostic pipeline")
    
    def run_analysis(self, landscape_steps=25, hessian_iters=20):
        """
        Runs the complete diagnostic analysis.
        
        Args:
            landscape_steps: Number of points in loss landscape
            hessian_iters: Number of power iterations for Hessian
        
        Returns:
            Dictionary of metrics for each model
        """
        criterion = nn.CrossEntropyLoss()
        
        print("\n" + "="*60)
        print("RUNNING QUANTIZATION DIAGNOSTICS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n[{name}] Starting analysis...")
            model.to(self.device)
            
            # 1. Hessian Spectrum (Sharpness)
            print(f"  → Computing Hessian eigenvalue (sharpness)...")
            sharpness = power_iteration(
                model, self.loader, criterion, self.device, hessian_iters
            )
            self.metrics[name] = {'sharpness': sharpness}
            print(f"  ✓ Top eigenvalue (λ_max): {sharpness:.6f}")
            
            # 2. Loss Landscape (Geometry)
            print(f"  → Tracing 1D loss landscape...")
            x, y = trace_1d_loss(
                model, self.loader, criterion, self.device, landscape_steps
            )
            self.traces[name] = (x, y)
            print(f"  ✓ Landscape traced ({landscape_steps} points)")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        self._print_summary()
        
        return self.metrics
    
    def _print_summary(self):
        """Prints a formatted summary of results"""
        print("\n📊 SHARPNESS COMPARISON:")
        for name, metrics in self.metrics.items():
            print(f"   {name:12s} λ_max = {metrics['sharpness']:.6f}")
        
        if len(self.metrics) > 1:
            fp32_sharp = self.metrics.get('FP32', {}).get('sharpness', 0)
            int8_sharp = self.metrics.get('Int8', {}).get('sharpness', 0)
            if fp32_sharp > 0:
                ratio = int8_sharp / fp32_sharp
                print(f"\n   Sharpness Ratio (Int8/FP32): {ratio:.2f}x")
                if ratio > 1.5:
                    print("   ⚠️  Quantization significantly increased sharpness!")
    
    def plot(self, save_path="bit_collapse.png", dpi=300):
        """
        Generates and saves the loss landscape visualization.
        
        Args:
            save_path: Path to save the figure
            dpi: Image resolution
        """
        if not self.traces:
            raise ValueError("No landscape data available. Run run_analysis() first.")
        
        print(f"\n📈 Generating visualization...")
        plot_overlay(self.traces, save_path, dpi)
