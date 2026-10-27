import torch

def function01(tensor: torch.Tensor, count_over: str) -> torch.Tensor:
    if count_over == 'columns':
        return tensor.mean(0)
    elif count_over == 'rows':
        return tensor.mean(1)
    else:
        raise ValueError

# a = torch.tensor([[0.8823, 0.9150, 0.3829],[0.9593, 0.3904, 0.6009],[0.2566, 0.7936, 0.9408]])
# b = torch.tensor([[0.8823, 0.9150, 0.3829],[0.9593, 0.3904, 0.6009],[0.2566, 0.7936, 0.9408]])

# print(function01(a, 'rows'))
# print(function01(b, 'columns'))