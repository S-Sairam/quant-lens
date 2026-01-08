import torch

def power_iteration(model, dataloader, criterion, device, num_iters=20):
    """
    Computes the top eigenvalue of the Hessian using power iteration.
    
    Power iteration algorithm:
    1. Start with random vector v
    2. Repeatedly compute v <- H*v / ||H*v||
    3. Eigenvalue λ ≈ v^T * H * v
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        criterion: Loss function
        device: 'cuda' or 'cpu'
        num_iters: Number of power iterations
    
    Returns:
        top_eigenvalue: Largest eigenvalue (sharpness metric)
    """
    model.eval()
    model.to(device)
    
    # Get a batch
    inputs, targets = next(iter(dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Initialize random vector
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]
    
    # Normalize
    v_norm = torch.sqrt(sum([torch.sum(vi**2) for vi in v]))
    v = [vi / v_norm for vi in v]
    
    for _ in range(num_iters):
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        output = model(inputs)
        loss = criterion(output, targets)
        
        # Compute gradient
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Hessian-vector product: H*v
        grad_v_product = sum([torch.sum(g * vi) for g, vi in zip(grads, v)])
        hv = torch.autograd.grad(grad_v_product, params, retain_graph=False)
        
        # Update v <- H*v
        v = [h.detach() for h in hv]
        
        # Normalize
        v_norm = torch.sqrt(sum([torch.sum(vi**2) for vi in v]))
        if v_norm > 1e-10:
            v = [vi / v_norm for vi in v]
    
    # Compute Rayleigh quotient: λ = v^T * H * v
    model.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_v_product = sum([torch.sum(g * vi) for g, vi in zip(grads, v)])
    hv = torch.autograd.grad(grad_v_product, params, retain_graph=False)
    
    eigenvalue = sum([torch.sum(h * vi) for h, vi in zip(hv, v)]).item()
    
    return eigenvalue

