import numpy as np
import torch

def get_random_direction(model):
    """
    Generates a random direction vector with filter normalization.
    
    Filter normalization (Li et al., 2018):
    d_{i,j} <- (d_{i,j} / ||d_{i,j}||) * ||θ_{i,j}||
    
    This ensures scale-invariance for networks with BatchNorm and ReLU.
    """
    direction = []
    for param in model.parameters():
        if param.requires_grad:
            d = torch.randn_like(param)
            # Filter-wise normalization
            if d.numel() > 0:
                d_norm = d.norm()
                p_norm = param.data.norm()
                if d_norm > 1e-10:
                    d = d * (p_norm / d_norm)
            direction.append(d)
        else:
            direction.append(None)
    return direction


def trace_1d_loss(model, loader, criterion, device, steps=25, distance=0.5):
    """
    Traces the loss landscape in a random direction.
    
    Args:
        model: PyTorch model
        loader: DataLoader (uses first batch for calibration)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: 'cuda' or 'cpu'
        steps: Number of points to sample
        distance: Maximum distance to walk in each direction
    
    Returns:
        alphas: Array of interpolation coefficients
        losses: Array of loss values
    """
    model.eval()
    model.to(device)
    
    # Get calibration batch
    try:
        inputs, targets = next(iter(loader))
    except StopIteration:
        raise ValueError("DataLoader is empty")
    
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Generate random direction with filter normalization
    direction = get_random_direction(model)
    
    # Save original weights
    orig_weights = [p.clone() if p is not None else None 
                    for p in model.parameters()]
    
    # Walk the landscape
    alphas = np.linspace(-distance, distance, steps)
    losses = []
    
    for alpha in alphas:
        # Perturb weights: θ_new = θ_0 + α * d
        for p, w0, d in zip(model.parameters(), orig_weights, direction):
            if p.requires_grad and w0 is not None and d is not None:
                p.data = w0 + alpha * d
        
        # Compute loss
        with torch.no_grad():
            output = model(inputs)
            loss = criterion(output, targets).item()
            losses.append(loss)
    
    # Restore original weights
    for p, w0 in zip(model.parameters(), orig_weights):
        if w0 is not None:
            p.data = w0.clone()
    
    return alphas, losses

