import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from multi_scale_model import get_multiscale_model

# STARC-9 Custom Modules
import sys
sys.path.append(os.path.abspath('STARC-9-Evaluation'))
from config import NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE

# A mock dataset representing how Multi-Scale Data must be loaded
class MockMultiScaleDataset(Dataset):
    def __init__(self, high_res_folder, low_res_folder, transform=None):
        """
        In real usage, you must map the high resolution 256x256 tumor tile 
        to its corresponding zoomed out 256x256 macro perspective.
        """
        self.transform = transform
        # Placeholders for data
        self.data_pairs = [] # List of tuples: (high_res_path, low_res_path, label)
        
        # Populate with dummy data for script verification
        self.data_pairs.append(("dummy1.jpg", "dummy1_macro.jpg", 8)) # 8 = Tumor
        self.data_pairs.append(("dummy2.jpg", "dummy2_macro.jpg", 5)) # 5 = Normal
        
    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        high_res_path, low_res_path, label = self.data_pairs[idx]
        
        # In a real environment, read the actual images
        # h_img = Image.open(high_res_path).convert('RGB')
        # l_img = Image.open(low_res_path).convert('RGB')
        
        # Use random tensors for testing the pipeline locally
        h_img = torch.randn(3, 256, 256)
        l_img = torch.randn(3, 256, 256)
        
        if self.transform:
            pass # Transforms are usually applied on PIL images before ToTensor
            
        return h_img, l_img, label

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Multiscale Training on: {device}")
    
    model = get_multiscale_model(num_classes=NUM_CLASSES)
    model.to(device)
    
    dataset = MockMultiScaleDataset("local_path", "global_path")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Quick Test Execution pass to verify the architecture
    for epoch in range(1): # Using 1 epoch to verify backward pass
        model.train()
        for batch_idx, (local_img, global_img, labels) in enumerate(dataloader):
            local_img = local_img.to(device)
            global_img = global_img.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Provide both image resolutions
            outputs = model(local_img, global_img)
            
            # Backward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            print(f"Dual-Branch Training Step - Loss: {loss.item():.4f}")
            break # Just testing one batch
            
    print("Multi-Scale Model verified successfully!")
    print("In order to proceed with actual training, the STARC-9 WSI slides must be extracted locally at a 5x and 20x zoom factor in parallel folders.")

if __name__ == "__main__":
    train()
