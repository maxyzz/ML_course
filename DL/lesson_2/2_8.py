import torch
import torch.nn as nn
from collections import OrderedDict


def create_model():
    model = nn.Sequential(
        nn.Linear(100, 10),  # Первый слой: 100 входов, 10 выходов
        nn.ReLU(),           # Функция нелинейности ReLU
        nn.Linear(10, 1)     # Второй слой: 10 входов, 1 выход
    )
    return model
