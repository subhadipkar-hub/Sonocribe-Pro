import torch.nn as nn
from torchvision import models

def build_model(num_classes=3):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Unfreeze all layers
    for param in model.features.parameters():
        param.requires_grad = True

    # Replace classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
