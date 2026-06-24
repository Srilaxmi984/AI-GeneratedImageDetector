"""
Week 1 - Step 2: Train a baseline CNN from scratch

Goal: 
- Build a simple CNN (no transfer learning yet)
- Train on sampled CIFAKE data
- Get baseline accuracy
- Save the trained model

This is your "before" metric. In Week 3, you'll add transfer learning
and compare: "simple CNN got X%, ResNet got Y%"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# =====================================================================
# PART 1: UNDERSTANDING THE ARCHITECTURE
# =====================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      BASELINE CNN ARCHITECTURE                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

A CNN learns to detect patterns in images through:

1. CONVOLUTION: Slides a filter over the image, extracting local patterns
   - Input: 32x32 RGB image (3 channels)
   - Filter size: 3x3 (learns edges, textures, etc.)
   - Output: Feature map showing where patterns appear

2. POOLING: Reduces size, keeps important info
   - MaxPooling: "Which pixel in this region is most important?"
   - Reduces computation, prevents overfitting

3. FLATTENING: Convert 2D feature maps to 1D vector

4. FULLY CONNECTED: Classify
   - Takes flattened features → outputs probabilities for each class
   - Class 0 = Real, Class 1 = Fake

Visual example (simplified):
Input Image (32×32×3)
    ↓
Conv2d(3→32) [learns 32 different filters]
    ↓
ReLU [activation - makes network non-linear]
    ↓
MaxPool2d [reduces 32×32 → 16×16]
    ↓
Conv2d(32→64) [learns 64 more complex filters]
    ↓
ReLU
    ↓
MaxPool2d [reduces 16×16 → 8×8]
    ↓
Flatten [8×8×64 = 4096 values]
    ↓
Linear(4096→128) [compress to 128 features]
    ↓
Linear(128→2) [output: probability for [Real, Fake]]
    ↓
Output: logits for 2 classes
""")

# =====================================================================
# PART 2: SIMPLE CNN ARCHITECTURE
# =====================================================================

class SimpleBaseCNN(nn.Module):
    """
    A simple CNN for binary classification (real vs fake images).
    
    Architecture:
    - 2 convolutional blocks (each: Conv2d → ReLU → MaxPool)
    - Flatten
    - 2 fully connected layers
    - Dropout for regularization (prevents overfitting)
    """
    
    def __init__(self):
        super(SimpleBaseCNN, self).__init__()
        
        # Block 1: Input (3 channels) → 32 filters
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 32×32 → 16×16
        
        # Block 2: 32 filters → 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 16×16 → 8×8
        
        # After pooling: 8×8×64 = 4096 features
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 4096 → 128
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Randomly drop 50% of neurons during training
        self.fc2 = nn.Linear(128, 2)  # 128 → 2 (Real or Fake)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Fully connected
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# =====================================================================
# PART 3: CUSTOM DATASET LOADER
# =====================================================================

class CIFAKESubset(Dataset):
    """
    Load images from real/ and fake/ directories.
    Label: 0 = Real, 1 = Fake
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load real images (label 0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for img_path in sorted(real_dir.glob("*.*")):
                self.images.append(img_path)
                self.labels.append(0)  # Real
        
        # Load fake images (label 1)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob("*.*")):
                self.images.append(img_path)
                self.labels.append(1)  # Fake
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms (resize, normalize, etc.)
        if self.transform:
            img = self.transform(img)
        
        return img, label


# =====================================================================
# PART 4: TRAIN AND EVALUATE
# =====================================================================

def train_baseline_cnn(data_dir="data/sampled_subset", epochs=10, batch_size=32):
    """
    Train the baseline CNN.
    
    Args:
        data_dir: Path to sampled CIFAKE data
        epochs: Number of training passes
        batch_size: Images per batch
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"✗ Data directory not found: {data_dir}")
        print("Run download_and_sample.py first!")
        return
    
    # =====================================================================
    # Data transforms (normalization, augmentation)
    # =====================================================================
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip images (real and fake both flip)
        transforms.RandomRotation(10),  # Small rotations
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # =====================================================================
    # Load full dataset
    # =====================================================================
    print(f"\nLoading data from: {data_dir}")
    full_dataset = CIFAKESubset(data_dir, transform=train_transform)
    print(f"Total images: {len(full_dataset)}")
    print(f"  Real: {sum(1 for l in full_dataset.labels if l == 0)}")
    print(f"  Fake: {sum(1 for l in full_dataset.labels if l == 1)}")
    
    # =====================================================================
    # Split: 60% train, 20% val, 20% test
    # =====================================================================
    total = len(full_dataset)
    train_size = int(0.6 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_dataset)} ({100*len(train_dataset)//total}%)")
    print(f"  Val:   {len(val_dataset)} ({100*len(val_dataset)//total}%)")
    print(f"  Test:  {len(test_dataset)} ({100*len(test_dataset)//total}%)")
    
    # =====================================================================
    # Data loaders (batch processing)
    # =====================================================================
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # =====================================================================
    # Initialize model, loss, optimizer
    # =====================================================================
    model = SimpleBaseCNN().to(device)
    criterion = nn.CrossEntropyLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate = 0.001
    
    print(f"\n{model}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # =====================================================================
    # Training loop
    # =====================================================================
    history = {"train_loss": [], "val_acc": [], "val_loss": []}
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(epochs):
        # --------- TRAINING PHASE ---------
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        
        # --------- VALIDATION PHASE ---------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Don't compute gradients during validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # =====================================================================
    # Test phase
    # =====================================================================
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f"\n✓ Test Accuracy: {test_acc:.2f}%")
    
    # =====================================================================
    # Save model
    # =====================================================================
    model_path = "models/baseline_cnn.pth"
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save metrics
    metrics = {
        "test_accuracy": test_acc,
        "val_accuracy": max(history["val_acc"]),
        "final_train_loss": history["train_loss"][-1],
    }
    with open("models/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: models/baseline_metrics.json")
    
    # =====================================================================
    # Plot training curves
    # =====================================================================
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Validation Accuracy over epochs")
    
    plt.tight_layout()
    plt.savefig("models/baseline_training_curves.png")
    print(f"✓ Training curves saved to: models/baseline_training_curves.png")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline CNN Test Accuracy: {test_acc:.2f}%")
    print(f"\nThis is your 'before' metric.")
    print(f"Next week: Train ResNet (transfer learning) and compare!")


if __name__ == "__main__":
    # Run training
    train_baseline_cnn(
        data_dir="data/sampled_subset",
        epochs=10,  # Adjust if needed (fewer = faster, but may underfit)
        batch_size=32  # Adjust for GPU memory (lower = less memory, slower)
    )
    
    print("\n✓ Week 1 complete!")
    print("Next: Run train_resnet_transfer_learning.py (Week 2-3)")