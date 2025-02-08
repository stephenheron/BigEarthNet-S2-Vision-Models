import torch
import torch.nn.functional as F
from torch import nn
from training_framework import BaseModel
from transformers import get_cosine_schedule_with_warmup

class ViT(BaseModel):
    def __init__(self, img_width, img_channels, patch_size, d_model, num_heads, num_layers, num_classes, ff_dim):
        super().__init__(num_classes)

        self.patch_size = patch_size

        # Patch embedding
        self.patch_embedding = nn.Linear(img_channels * patch_size * patch_size, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final classification layer
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        N, C, H, W = x.shape
        # Patch division and flattening
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(N, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, C * self.patch_size * self.patch_size)
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Transform
        x = self.transformer_encoder(x)
        
        # Get CLS token output
        x = x[:, 0]
        
        # Final classification
        x = self.fc(x)
        return x
    
    def get_learning_rate_scheduler(self, optimizer, steps_per_epoch, num_epochs):
        num_training_steps = num_epochs * steps_per_epoch
        num_warmup_steps = num_training_steps * 0.05
        
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

