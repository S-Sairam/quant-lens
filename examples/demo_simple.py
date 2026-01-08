import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from quant_lens import QuantDiagnostic

# Create simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

# Create dummy data
X = torch.randn(200, 10)
y = torch.randint(0, 10, (200,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32)

# Run diagnostic
diagnostic = QuantDiagnostic(model, loader, device='cpu')
diagnostic.add_int8_model()
metrics = diagnostic.run_analysis(landscape_steps=15, hessian_iters=10)
diagnostic.plot(save_path='simple_demo.png')

print("\n✅ Demo complete! Check simple_demo.png")
