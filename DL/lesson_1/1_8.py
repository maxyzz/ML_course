import torch

def function04(x: torch.Tensor, y: torch.Tensor):
    layer = torch.nn.Linear(x.shape[1], 1, bias=False)
    step_size = 1e-2
    mse = 2
    # i=0
    while mse > 0.3:
        y_pred = layer(x).ravel()
        
        mse = torch.mean((y_pred - y) ** 2)

        mse.backward()
        
        with torch.no_grad():
            layer.weight.data -= layer.weight.grad * step_size

        layer.zero_grad()
        
    return layer

# n_features = 2
# n_objects = 300

# w_true = torch.randn(n_features)
# X = (torch.rand(n_objects, n_features) - 0.5) * 5
# Y = X @ w_true + torch.randn(n_objects) / 2

# print(w_true)
# layer = function04(X, Y)
# print("Обученные веса слоя:", layer.weight.data)