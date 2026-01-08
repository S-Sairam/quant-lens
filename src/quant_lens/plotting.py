import matplotlib.pyplot as plt
import torch


def plot_overlay(traces_dict, save_path="loss_landscape.png", dpi=300):
    """
    Creates an overlay plot of multiple loss landscapes.
    
    Args:
        traces_dict: Dict mapping model names to (x, y) tuples
        save_path: Path to save the figure
        dpi: Image resolution
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D00000']
    
    for idx, (name, (x, y)) in enumerate(traces_dict.items()):
        color = colors[idx % len(colors)]
        ax.plot(x, y, label=name, linewidth=2.5, color=color, alpha=0.85)
    
    ax.set_xlabel('α (Interpolation Coefficient)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Landscape: FP32 vs Quantized Models', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Plot saved to: {save_path}")
    plt.close()
