import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

@torch.inference_mode()
def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    # 1. Переведите модель в режим инференса (применения)
    model.eval()
    
    total_loss = 0.0
    
    # 2. Проитерируемся по даталоадеру
    for batch in data_loader:
        x, y = batch
        
        # 3. На каждой итерации:

            # - Сделайте проход вперед (forward pass)
        y_pred = model(x)
        
        # - Посчитайте ошибку
        loss = loss_fn(y_pred, y)
        
        
        # Накапливаем ошибку для подсчета среднего
        total_loss += loss.item()
        
    # Функция должна вернуть среднюю ошибку за время прохода по даталоадеру
    return total_loss / len(data_loader)