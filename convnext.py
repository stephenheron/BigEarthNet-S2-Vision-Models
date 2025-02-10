import torch
import torch.nn.functional as F
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torch import nn
from training_framework import BaseModel
from transformers import get_cosine_schedule_with_warmup

class ConvNeXt(BaseModel):
    def __init__(self, img_channels, num_classes):
        super().__init__(num_classes)

        # Load pretrained ConvNeXt-Tiny
        self.model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)

        # Modify first convolution layer to accept 5 channels
        # Get weights of original conv1 layer
        original_conv = self.model.features[0][0]
        original_weights = original_conv.weight
        
        # Create new conv layer with 5 input channels
        new_conv = nn.Conv2d(
            in_channels=img_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            groups=original_conv.groups,
            bias=original_conv.bias is not None,
            dilation=original_conv.dilation
        )

        # For the first 3 channels, use pretrained weights
        new_conv.weight.data[:, :3, :, :] = original_weights

        # For the additional channels, initialize with mean of RGB channels
        channel_mean = original_weights.mean(dim=1, keepdim=True)
        new_conv.weight.data[:, 3:, :, :] = channel_mean.repeat(1, img_channels-3, 1, 1)

         # Replace first conv layer
        self.model.features[0][0] = new_conv

         # Modify classifier for 120x120 input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, img_channels, 120, 120)
            features = self.model.features(dummy_input)
        
        avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            avgpool,
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(features.shape[1], num_classes)
        )

        # Freeze/unfreeze layers based on stages
        self._freeze_layers()

    def _freeze_layers(self):

        '''
        ConvNeXt-Tiny architecture layout from the model:
        features[0] - First conv + norm
        features[1] - Stage 0 (3 CNBlocks)
        features[2] - Downsample to stage 1
        features[3] - Stage 1 (3 CNBlocks)
        features[4] - Downsample to stage 2
        features[5] - Stage 2 (9 CNBlocks)
        features[6] - Downsample to stage 3
        features[7] - Stage 3 (3 CNBlocks)
        '''

        # First, freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # First conv layer
        for param in self.model.features[0].parameters():
            param.requires_grad = True

        # Unfreze layer 2 and 3
        for param in self.model.features[5].parameters():
            param.requires_grad = True
        for param in self.model.features[7].parameters():
            param.requires_grad = True

         # Always unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        features = self.model.features(x)
        out = self.classifier(features)
        return out
    
    def get_learning_rate_scheduler(self, optimizer, steps_per_epoch, num_epochs):
        num_training_steps = num_epochs * steps_per_epoch
        num_warmup_steps = num_training_steps * 0.05
        
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

