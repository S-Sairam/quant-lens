import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class FakeQuantOp(torch.autograd.Function):
    """
    Fake Quantization with Straight-Through Estimator (STE).
    Forward: Simulates int8 quantization (bit collapse)
    Backward: Passes gradients through (identity function)
    """
    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmin = -2**(num_bits-1)
        qmax = 2**(num_bits-1) - 1
        scale = x.abs().max() / qmax if x.abs().max() > 0 else 1.0
        x_quant = (x / scale).round().clamp(qmin, qmax)
        return x_quant * scale

    @staticmethod
    def backward(ctx, grad_output):
        # STE: gradient flows straight through
        return grad_output, None


class QuantLinear(nn.Linear):
    """Linear layer with fake quantized weights"""
    def __init__(self, in_features, out_features, bias=True, num_bits=8):
        super().__init__(in_features, out_features, bias)
        self.num_bits = num_bits
    
    def forward(self, input):
        w_q = FakeQuantOp.apply(self.weight, self.num_bits)
        return F.linear(input, w_q, self.bias)


def replace_linear_layers(model, num_bits=8):
    """
    Recursively replaces all nn.Linear layers with QuantLinear.
    Creates a FakeQuant version of the model for landscape analysis.
    """
    model_q = copy.deepcopy(model)
    
    def recursive_replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # Replace with QuantLinear
                new_layer = QuantLinear(
                    child.in_features, 
                    child.out_features, 
                    child.bias is not None,
                    num_bits
                )
                # Copy weights
                new_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()
                setattr(module, name, new_layer)
            else:
                # Recurse into nested modules
                recursive_replace(child)
    
    recursive_replace(model_q)
    return model_q
