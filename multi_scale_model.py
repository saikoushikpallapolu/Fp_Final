import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MultiScaleCRCNet(nn.Module):
    def __init__(self, num_classes=9, pretrained=True):
        super(MultiScaleCRCNet, self).__init__()
        
        # Branch 1: High Resolution (e.g. 256x256 local crop at 20x zoom)
        # Using ResNet50 feature extractor
        self.local_branch = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        # Remove the final fully connected layer to output raw 2048-dim features
        self.local_branch.fc = nn.Identity() 

        # Branch 2: Low Resolution (e.g. 256x256 macro crop resized from 1024x1024 at 5x zoom)
        self.global_branch = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.global_branch.fc = nn.Identity()
        
        # Fusion Classifier 
        # Feature sizes: Resnet50 typical pooled output is 2048. 
        # Concatenated branch output will be 4096.
        self.fusion = nn.Sequential(
            nn.Linear(2048 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_local, x_global):
        """
        Forward pass expects a tuple of two images natively representing scales
        :param x_local: Tensor (B, C, H, W) of the zoomed-in high mag image
        :param x_global: Tensor (B, C, H, W) of the zoomed-out low mag image
        """
        # Extract individual feature embeddings
        features_local = self.local_branch(x_local)   # Shape: [B, 2048]
        features_global = self.global_branch(x_global) # Shape: [B, 2048]
        
        # Merge contexts (Dual-scale learning)
        features_merged = torch.cat((features_local, features_global), dim=1) # Shape: [B, 4096]
        
        # Output unscaled logits
        logits = self.fusion(features_merged) # Shape: [B, num_classes]
        return logits

def get_multiscale_model(num_classes=9):
    """Factory function designed similarly to the STARC-9 existing system."""
    model = MultiScaleCRCNet(num_classes=num_classes)
    return model
