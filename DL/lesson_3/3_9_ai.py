import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets, transforms

def create_mlp_model() -> nn.Module:
    """
    Создает многослойный перцептрон (MLP) для классификации MNIST.
    """
    model = nn.Sequential(
        nn.Flatten(),                      # Превращает 28x28 в 784
        nn.Linear(784, 512),               # Первый скрытый слой
        nn.ReLU(),
        nn.BatchNorm1d(512),               # Ускоряет сходимость и стабилизирует обучение
        nn.Dropout(0.2),                   # Защита от переобучения
        
        nn.Linear(512, 256),               # Второй скрытый слой
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        
        nn.Linear(256, 10)                 # Выходной слой (10 классов)
    )
    return model

def train(
    model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn
) -> tuple[float, float]:
    model.train()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(data_loader, desc='Train'):
        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)

        loss.backward()

        total_loss += loss.item()

        optimizer.step()

        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()

    return total_loss / len(data_loader), correct / total

def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn) -> tuple[float, float]:
    model.eval()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(data_loader, desc='Evaluate'):
        output = model(x)

        loss = loss_fn(output, y)

        total_loss += loss.item()

        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()

    return total_loss / len(data_loader), correct / total


if __name__ == '__main__':
    # Подготовка датасета с нормализацией
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Стандартное среднее и отклонение для MNIST
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Инициализация
    model = create_mlp_model()
    criterion = nn.CrossEntropyLoss()
    # AdamW и небольшой weight_decay помогают стабильно пробить порог в 98%
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Цикл обучения (обычно достаточно 3-5 эпох)
    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        print(f"Эпоха {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
        
        # Если цель в 98% достигнута, сохраняем веса
        if test_acc >= 0.98:
            print("Целевая метрика в 98% достигнута!")
            torch.save(model.state_dict(), 'weights.pt')
            print("Веса успешно сохранены в файл 'weights.pt'")
            break