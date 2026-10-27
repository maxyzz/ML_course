import torch

def function03(x: torch.Tensor, y: torch.Tensor):
    num_features = x.shape[1]
    weights = torch.rand(num_features, dtype=torch.float32, requires_grad=True)
    step_size = 1e-2
    mse = 2
    # i=0
    while mse > 1:
        y_pred = torch.matmul(x, weights) 
        
        mse = torch.mean((y_pred - y) ** 2)

        mse.backward()
        
        with torch.no_grad():
            weights.data -= weights.grad * step_size

        # print(f"\r i = {i}, mse = {mse}", end="", flush=True)
        weights.grad.zero_()
        # i+=1
    # print()
    return weights

# n_features = 2         # Размерность указана для примера, не нужно использовать её в коде
# n_objects = 300

# w_true = torch.randn(n_features)
# X = (torch.rand(n_objects, n_features) - 0.5) * 5
# Y = X @ w_true + torch.randn(n_objects) / 2

# print(w_true)
# weights = function03(X, Y)
# print(weights)