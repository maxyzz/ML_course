import torch
from torchvision.models import alexnet, vgg11, googlenet, resnet18

def get_pretrained_model(model_name: str, num_classes: int, pretrained: bool = True):
    # model_name: 'alexnet', 'vgg11', 'googlenet', 'resnet18'
    
    # 1. Загружаем базовую модель напрямую
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        
    elif model_name == 'alexnet':
        model = alexnet(pretrained=pretrained)
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes)
        
    elif model_name == 'vgg11':
        model = vgg11(pretrained=pretrained)
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes)
        
    elif model_name == 'googlenet':
        model = googlenet(pretrained=pretrained, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        
        # Безопасная адаптация вспомогательных выходов aux1 и aux2
        for aux_layer_name in ['aux1', 'aux2']:
            if hasattr(model, aux_layer_name):
                aux_layer = getattr(model, aux_layer_name)
                if aux_layer is not None:
                    # В новых версиях слой называется 'fc', в старых — 'fc2'
                    attr_name = 'fc' if hasattr(aux_layer, 'fc') else 'fc2'
                    linear_layer = getattr(aux_layer, attr_name)
                    
                    # Заменяем слой на новый с нужным количеством классов
                    setattr(aux_layer, attr_name, torch.nn.Linear(linear_layer.in_features, num_classes))
            
    else:
        raise ValueError(f"Модель {model_name} не поддерживается этой функцией.")
        
    return model
