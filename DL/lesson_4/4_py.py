import torch

# def get_normalize(features: torch.Tensor):
#     # 1. Считаем среднее по каналам
#     mean = features.mean(dim=(0, 2, 3))
    
#     # 2. Считаем стандартное отклонение по каналам (несмещенное по умолчанию в PyTorch)
#     std = features.std(dim=(0, 2, 3), correction=1)
    
#     return mean, std


# import torchvision.transforms as T

# def get_augmentations(train: bool = True) -> T.Compose:
#     # Константы нормализации для CIFAR10, указанные на подсказке ко 2-му скриншоту
#     means = (0.49139968, 0.48215841, 0.44653091)
#     stds = (0.24703223, 0.24348513, 0.26158784)
    
#     if train:
#         # Аугментации для обучающей выборки (с добавлением случайного отражения)
#         return T.Compose([
#             T.Resize((224, 224)),
#             T.RandomHorizontalFlip(), # Любая базовая аугментация с занятия
#             T.ToTensor(),
#             T.Normalize(mean=means, std=stds)
#         ])
#     else:
#         # Аугментации для тестовой/валидационной выборки (без случайных изменений)
#         return T.Compose([
#             T.Resize((224, 224)),
#             T.ToTensor(),
#             T.Normalize(mean=means, std=stds)
#         ])

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
#     # Переводим модель в режим оценки (выключает Dropout и BatchNorm)
#     model.eval()
#     model.to(device)
    
#     # 1. Создаем пустой список для хранения предсказаний
#     all_predictions = []
    
#     # Отключаем подсчет градиентов для ускорения инференса
#     with torch.no_grad():
#         # 2. Проитерируемся по даталоадеру
#         for x, _ in loader:
#             # Переносим входной батч картинок на целевое устройство (процессор или видеокарту)
#             x = x.to(device)
            
#             # 3. Делаем forward pass для батча
#             logits = model(x)
            
#             # Считаем классы как аргмакс по выходу (вдоль размерности классов dim=1)
#             y_pred = torch.argmax(logits, dim=1)
            
#             # Добавляем тензор с предсказаниями в список
#             all_predictions.append(y_pred)
            
#     # 4. Делаем конкатенацию всех предсказаний в один длинный тензор
#     # Переносим итоговый тензор на CPU (обычно хорошая практика для чекеров, 
#     # но если чекер требует оставить на девайсе, можно .cpu() убрать)
#     final_predictions = torch.cat(all_predictions, dim=0)
    
#     return final_predictions

# def create_simple_conv_cifar() -> nn.Module:
#     """
#     Создает модель с ровно двумя сверточными слоями для CIFAR10.
#     """
#     model = nn.Sequential(
#         # Блок свертки 1
#         nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), # Вход: 3x32x32
#         nn.BatchNorm2d(32),   # Стабилизирует обучение
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2), # Выход: 32x16x16
#         nn.Dropout2d(0.1),

#         # Блок свертки 2 (последний сверточный по ограничению)
#         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2), # Выход: 64x8x8
#         nn.Dropout2d(0.2),

#         # Классификатор (Полносвязные слои)
#         nn.Flatten(), # Выпрямляем: 64 * 8 * 8 = 4096
#         nn.Linear(64 * 8 * 8, 256),
#         nn.BatchNorm1d(256),
#         nn.ReLU(),
#         nn.Dropout(0.3),
#         nn.Linear(256, 10) # 10 классов CIFAR10
#     )
#     return model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@torch.no_grad()
def predict_tta(model: nn.Module, loader: DataLoader, device: torch.device, iterations: int = 2) -> torch.Tensor:
    # Переводим модель в режим инференса
    model.eval()
    model.to(device)
    
    # Список, где будем хранить итоговые тензоры логитов размером [N, C] для каждой итерации
    iteration_logits = []
    
    # 1. Запускаем цикл по количеству итераций
    for _ in range(iterations):
        single_prediction = []
        
        # 2. Внутри цикла проитерируемся по даталоадеру
        for x, _ in loader:
            x = x.to(device)
            
            # Получаем сырые выходы нейросети (логиты)
            logits = model(x)
            single_prediction.append(logits)
            
        # 3. Записываем ответы в один большой тензор размера [N, C] для текущей итерации
        # Используем конструкцию по подсказке ко второму скриншоту
        full_iteration_logits = torch.vstack(single_prediction)
        iteration_logits.append(full_iteration_logits)
        
    # 4. Собираем тензоры всех итераций в один огромный тензор
    # Сначала упаковываем в [iterations, N, C], затем меняем оси на [N, C, iterations]
    stacked_logits = torch.stack(iteration_logits, dim=0) # Размерность: [iterations, N, C]
    permuted_logits = stacked_logits.permute(1, 2, 0)     # Размерность: [N, C, iterations]
    
    # Усредняем его по оси итераций (dim=2), получая тензор размера [N, C]
    mean_logits = permuted_logits.mean(dim=2)
    
    # 5. Предсказываем классы как аргмакс по выходам вдоль оси классов (dim=1)
    final_classes = torch.argmax(mean_logits, dim=1)
    
    return final_classes