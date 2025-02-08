import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
from training_framework import BaseModel
from transformers import get_cosine_schedule_with_warmup

class CNN(BaseModel):
    def __init__(self, img_channels, num_classes):
        super().__init__(num_classes)

         # Load pretrained ResNet
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Replace the stem with custom number of input channels
        self.custom_stem = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) 
        
        # Get all layers after the stem
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.freeze(self.layer1)
        self.freeze(self.layer2)
        
        # Adjust avgpool for new input size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification layer
        self.fc = nn.Linear(512, self.num_classes)
        
    def forward(self, x):
        x = self.custom_stem(x)        # Output: 120x120
        x = self.layer1(x)             # Output: 60x60
        x = self.layer2(x)             # Output: 30x30
        x = self.layer3(x)             # Output: 15x15
        x = self.layer4(x)             # Output: 8x8
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def freeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def get_learning_rate_scheduler(self, optimizer, steps_per_epoch, num_epochs):
        num_training_steps = num_epochs * steps_per_epoch
        num_warmup_steps = num_training_steps * 0.05
        
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )