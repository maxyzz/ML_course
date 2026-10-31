import torch.nn as nn
class AdvancedSkipConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- БЛОК 1: 32x32 -> 16x16 ---
        self.b1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.b1_bn1 = nn.BatchNorm2d(64)
        self.b1_relu1 = nn.ReLU()
        
        # Слой для skip-connection (вход 64 -> выход 64)
        self.b1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b1_bn2 = nn.BatchNorm2d(64)
        self.b1_relu2 = nn.ReLU()
        
        self.b1_pool = nn.MaxPool2d(2)
        self.b1_drop = nn.Dropout(0.2)

        # --- БЛОК 2: 16x16 -> 8x8 ---
        self.b2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.b2_bn1 = nn.BatchNorm2d(128)
        self.b2_relu1 = nn.ReLU()
        
        # Слой для skip-connection (вход 128 -> выход 128)
        self.b2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.b2_bn2 = nn.BatchNorm2d(128)
        self.b2_relu2 = nn.ReLU()
        
        self.b2_pool = nn.MaxPool2d(2)
        self.b2_drop = nn.Dropout(0.3)

        # --- БЛОК 3: 8x8 -> 4x4 ---
        self.b3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b3_bn1 = nn.BatchNorm2d(256)
        self.b3_relu1 = nn.ReLU()
        
        # Слой для skip-connection (вход 256 -> выход 256)
        self.b3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.b3_bn2 = nn.BatchNorm2d(256)
        self.b3_relu2 = nn.ReLU()
        
        self.b3_pool = nn.MaxPool2d(2)
        self.b3_drop = nn.Dropout(0.4)

        # --- Классификатор ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc_relu = nn.ReLU()
        self.fc_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Блок 1
        x = self.b1_relu1(self.b1_bn1(self.b1_conv1(x)))
        # Добавляем skip-connection: x = block(x) + x
        x = self.b1_relu2(self.b1_bn2(self.b1_conv2(x))) + x
        x = self.b1_drop(self.b1_pool(x))

        # Блок 2
        x = self.b2_relu1(self.b2_bn1(self.b2_conv1(x)))
        # Добавляем skip-connection
        x = self.b2_relu2(self.b2_bn2(self.b2_conv2(x))) + x
        x = self.b2_drop(self.b2_pool(x))

        # Блок 3
        x = self.b3_relu1(self.b3_bn1(self.b3_conv1(x)))
        # Добавляем skip-connection
        x = self.b3_relu2(self.b3_bn2(self.b3_conv2(x))) + x
        x = self.b3_drop(self.b3_pool(x))

        # Выпрямление и классификация
        x = self.flatten(x)
        x = self.fc_drop(self.fc_relu(self.fc_bn(self.fc1(x))))
        x = self.fc2(x)
        return x

def create_advanced_skip_connection_conv_cifar() -> nn.Module:
    """Обязательная функция по условию задачи, возвращающая модель."""
    return AdvancedSkipConvNet()