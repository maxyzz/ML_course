import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    # 1. Переводим модель в режим обучения
    model.train()
    
    total_loss = 0.0
    
    # 2. Проитерируемся по даталоадеру
    for batch in data_loader:
        x, y = batch
        
        # 3. На каждой итерации:
        # - Зануляем градиенты с помощью оптимизатора
        optimizer.zero_grad()
        
        # - Сделайте проход вперед (forward pass)
        y_pred = model(x)
        
        # - Посчитайте ошибку
        loss = loss_fn(y_pred, y)
        
        # - Сделайте проход назад (backward pass)
        loss.backward()
        
        # - Напечатайте ошибку на текущем батче с точностью до 5 символов после запятой (только число)
        print(f"{loss.item():.5f}")
        
        # Накапливаем ошибку для подсчета среднего
        total_loss += loss.item()
        
        # - Сделайте шаг оптимизации
        optimizer.step()
        
    # Функция должна вернуть среднюю ошибку за время прохода по даталоадеру
    return total_loss / len(data_loader)