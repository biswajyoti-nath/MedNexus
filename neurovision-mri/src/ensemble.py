import torch
import torch.nn as nn

class ClassifierEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        """
        models: List of loaded PyTorch models (classifiers)
        weights: List of weights for soft voting
        """
        super(ClassifierEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
            
    def forward(self, x):
        outputs = []
        for model in self.models:
            out = model(x)
            outputs.append(torch.sigmoid(out))
            
        # Weighted soft voting
        final_prob = torch.zeros_like(outputs[0])
        for prob, w in zip(outputs, self.weights):
            final_prob += prob * w
            
        return final_prob


class SegmentorEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        """
        models: List of loaded PyTorch models (segmentors)
        weights: List of weights for mask averaging
        """
        super(SegmentorEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
            
    def forward(self, x):
        outputs = []
        for model in self.models:
            out = model(x)
            outputs.append(torch.sigmoid(out))
            
        # Average probability mask
        final_mask = torch.zeros_like(outputs[0])
        for mask, w in zip(outputs, self.weights):
            final_mask += mask * w
            
        return final_mask
