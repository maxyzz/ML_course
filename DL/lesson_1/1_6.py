import torch

def function02(tensor: torch.Tensor) -> torch.Tensor:
    num_features = tensor.shape[1]
    weights = torch.rand(num_features, dtype=torch.float32, requires_grad=True)
    return weights
