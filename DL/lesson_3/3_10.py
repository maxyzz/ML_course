# import torch
import torch.nn as nn
# from collections import OrderedDict
# from torch.utils.data import DataLoader
# from torch.optim import Optimizer
# from tqdm import tqdm
# import torch.optim as optim
# from torchvision import datasets, transforms
# from time import perf_counter
# from torchvision.datasets import MNIST # Класс для загрузки датасета MNIST из TorchVision.
# import torchvision.transforms as T # Модуль TorchVision, предоставляющий стандартные преобразования для изображений.
# from IPython.display import clear_output
# import os

def create_conv_model() -> nn.Module:

    model = nn.Sequential(
        # Первый сверточный блок
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5), # Вход: 1 канал (MNIST), 32 выходных канала, ядро 5x5.
                                                                # Размер выхода: (28-5+1) = 24. Тензор: (N, 32, 24, 24).
        nn.ReLU(),                                                # Функция активации ReLU.
        nn.MaxPool2d(kernel_size=2),                              # Макс-пулинг 2x2. Уменьшает размер в 2 раза.
                                                                # Размер выхода: 24/2 = 12. Тензор: (N, 32, 12, 12).

        # Второй сверточный блок
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), # Вход: 32 канала (от предыдущего слоя), 64 выходных канала, ядро 5x5.
                                                                # Размер выхода: (12-5+1) = 8. Тензор: (N, 64, 8, 8).
        nn.ReLU(),                                                # Функция активации ReLU.
        nn.MaxPool2d(kernel_size=2),                              # Макс-пулинг 2x2. Уменьшает размер в 2 раза.
                                                                # Размер выхода: 8/2 = 4. Тензор: (N, 64, 4, 4).

        # Полносвязные слои для классификации
        nn.Flatten(),                                             # Разворачивает тензор (N, 64, 4, 4) в (N, 64*4*4=1024) вектор.
        nn.Linear(4 * 4 * 64, 256),                               # Первый полносвязный слой: 1024 входа, 256 выходов.
        nn.ReLU(),                                                # Функция активации ReLU.
        nn.Linear(256, 10)                                        # Второй полносвязный слой: 256 входов, 10 выходов (по числу классов).
    )
    
    # model = nn.Sequential(
    #         nn.Flatten(),                     # Разворачивает входное изображение 1x28x28 в вектор 784.
    #         nn.Linear(28 * 28, 256),          # Первый полносвязный слой: 784 входа, 256 выходов.
    #         nn.ReLU(),                        # Функция активации ReLU.
    #         nn.Linear(256, 10)                # Второй полносвязный слой: 256 входов, 10 выходов (по количеству классов MNIST).
    #     )
    return model

# def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn) -> tuple[float, float]:
#     model.train()

#     total_loss = 0
#     total = 0
#     correct = 0

#     for x, y in tqdm(data_loader, desc='Train'):
#         optimizer.zero_grad()

#         output = model(x)

#         loss = loss_fn(output, y)

#         loss.backward()

#         total_loss += loss.item()

#         optimizer.step()

#         _, y_pred = torch.max(output, 1)
#         total += y.size(0)
#         correct += (y_pred == y).sum().item()

#     return total_loss / len(data_loader), correct / total

# def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn) -> tuple[float, float]:
#     model.eval()

#     total_loss = 0
#     total = 0
#     correct = 0

#     for x, y in tqdm(data_loader, desc='Evaluate'):
#         output = model(x)

#         loss = loss_fn(output, y)

#         total_loss += loss.item()

#         _, y_pred = torch.max(output, 1)
#         total += y.size(0)
#         correct += (y_pred == y).sum().item()

#     return total_loss / len(data_loader), correct / total


# if __name__ == '__main__':
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     mnist_path = os.path.join(script_dir, "mnist")
#     weights_path = os.path.join(script_dir, "weights.pt")

#     model = create_conv_model()
#     criterion = nn.CrossEntropyLoss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     loss_fn = nn.CrossEntropyLoss()

#     num_epochs = 15 # Количество эпох обучения. Эпоха - это один полный проход по всей обучающей выборке.

#     train_loss_history, valid_loss_history = [], [] # Списки для хранения истории потерь на обучении и валидации.
#     valid_accuracy_history = []                     # Список для хранения истории точности на валидации.

#     start = perf_counter() # Записываем время начала обучения для измерения общей продолжительности.
#     mnist_train = MNIST(
#         mnist_path,
#         train=True,
#         download=True,
#         transform=T.ToTensor() # Применяем T.ToTensor для преобразования PIL Image в PyTorch Tensor
#     )
#     mnist_valid = MNIST(
#         mnist_path,
#         train=False, # Загружаем тестовую выборку
#         download=True,
#         transform=T.ToTensor()
#     )
#     # Создаем DataLoader для обучающей выборки.
#     # batch_size=64: Каждый батч будет содержать 64 изображения.
#     # shuffle=True: Перемешиваем данные в каждой эпохе.
#     train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

#     # Создаем DataLoader для валидационной выборки.
#     # shuffle=False: Для валидации порядок данных не имеет значения.
#     valid_loader = DataLoader(mnist_valid, batch_size=64, shuffle=False)

#     for epoch in range(num_epochs): # Запускаем цикл обучения на заданное количество эпох.
#         # Выполняем шаг обучения на обучающей выборке и получаем среднюю ошибку.
#         train_loss, train_acc = train(model,train_loader, optimizer, loss_fn)

#         # Оцениваем модель на валидационной выборке, получаем среднюю ошибку и точность.
#         valid_loss, valid_accuracy = evaluate(model, valid_loader, loss_fn)

#         # Добавляем полученные значения в списки истории.
#         train_loss_history.append(train_loss)
#         valid_loss_history.append(valid_loss)
#         valid_accuracy_history.append(valid_accuracy)
        
#         print(f"Эпоха {epoch}/{num_epochs} | "
#                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
#                       f"Test Loss: {valid_loss:.4f}, Test Acc: {valid_accuracy*100:.2f}%")
        
#         if valid_accuracy >= 0.993:
#             print("Целевая метрика в 98% достигнута!")
#             # torch.save(model.state_dict(), weights_path)
#             torch.save(model.state_dict(), weights_path, _use_new_zipfile_serialization=True)
#             print("Веса успешно сохранены в файл 'weights.pt'")
#             break

        
#     # Выводим общее время, затраченное на обучение и оценку.
#     print(f'Total training and evaluation time {perf_counter() - start:.5f} seconds')

