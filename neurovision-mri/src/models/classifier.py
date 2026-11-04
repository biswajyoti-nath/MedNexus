import torch
import torch.nn as nn
import timm

class BrainClassifier(nn.Module):
    def __init__(self, model_name="resnet34", pretrained=True):
        super(BrainClassifier, self).__init__()
        
        self.model_name = model_name
        
        # Load pretrained backbone from timm
        # num_classes=0 means we drop the classification head and just get features
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        
        # Determine the number of features output by the backbone
        if "resnet" in model_name:
            in_features = self.backbone.num_features
        elif "efficientnet" in model_name:
            in_features = self.backbone.num_features
        elif "densenet" in model_name:
            in_features = self.backbone.num_features
        else:
            # Fallback for timm models
            in_features = self.backbone.num_features
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features, 1)
        
    def forward(self, x):
        # Extract features (B, C, H, W)
        features = self.backbone(x)
        
        # Pool to (B, C, 1, 1) and flatten to (B, C)
        pooled = self.global_pool(features).flatten(1)
        
        # Classification head
        out = self.dropout(pooled)
        out = self.fc(out)
        
        return out

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
