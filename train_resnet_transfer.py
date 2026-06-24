"""
Week 2: Transfer Learning with ResNet-50

Goal:
- Load pretrained ResNet-50 (trained on ImageNet with 1M images)
- Fine-tune on CIFAKE data
- Compare: Baseline CNN (84.6%) vs ResNet (expected 92-95%)
- Show the power of transfer learning

Why ResNet?
- ResNet-50 = 50 layers deep, learned on ImageNet
- Early layers: edges, textures (we keep these frozen)
- Late layers: object features (we retrain these on CIFAKE)
- Result: Much better performance with less data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from PIL import Image

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   TRANSFER LEARNING WITH RESNET-50                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

What is Transfer Learning?

Normal approach (Baseline CNN):
  Random init → Learn everything from scratch → Result: OK

Transfer Learning:
  Load pretrained ImageNet weights → Fine-tune last layers → Result: Much better!

Why it works:
- Early layers learn universal patterns (edges, textures, colors)
- These patterns are useful for ANY image task
- ResNet-50 spent millions of GPU hours learning on ImageNet
- We reuse that knowledge instead of starting from zero

Architecture:
  ImageNet pretrained ResNet-50 (trained on 1M images)
    ↓
  Freeze early layers (conv1, conv2_x, conv3_x, conv4_x)
    ↓
  Replace final layer: 1000 classes → 2 classes (Real/Fake)
    ↓
  Fine-tune: Only retrain conv5_x + classification layer
    ↓
  Result: Fast training, better accuracy

Expected accuracy: 92-95% (vs baseline 84.6%)
Expected time: 3-8 mins on GPU
""")

# =====================================================================
# PART 1: DATASET (same as Week 1)
# =====================================================================

class CIFAKESubset(Dataset):
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
                self.labels.append(0)
        
        # Load fake images (label 1)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob("*.*")):
                self.images.append(img_path)
                self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # If image fails to load, return a blank image
            img = Image.new('RGB', (32, 32))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# =====================================================================
# PART 2: RESNET-50 WITH TRANSFER LEARNING
# =====================================================================

def create_resnet_model():
    """
    Load pretrained ResNet-50 and modify for binary classification.
    
    Key strategy:
    1. Load pretrained weights from ImageNet
    2. Freeze early layers (they already learned useful patterns)
    3. Replace final classification layer: 1000 → 2 classes
    4. Only fine-tune the last layers
    """
    
    # Load pretrained ResNet-50
    model = models.resnet50(pretrained=True)  # Download ~100MB weights
    
    print("\n✓ Loaded pretrained ResNet-50 from ImageNet")
    print(f"  - Original final layer: 1000 classes")
    print(f"  - We're changing to: 2 classes (Real/Fake)")
    
    # Strategy 1: Freeze early layers (don't train them)
    # This saves computation and prevents overfitting on small dataset
    for param in model.layer1.parameters():
        param.requires_grad = False  # Don't update these weights
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    
    # Only layer4 (last layers) and fc (classification) will be trained
    
    # Strategy 2: Replace final classification layer
    # ResNet outputs to a 1000-class layer (ImageNet has 1000 classes)
    # We need 2 classes (Real/Fake)
    num_features = model.fc.in_features  # Get input size to final layer
    model.fc = nn.Linear(num_features, 2)  # Replace with 2-class layer
    
    print(f"\n✓ Model modified:")
    print(f"  - Frozen layers: layer1, layer2, layer3 (early patterns)")
    print(f"  - Trainable layers: layer4, fc (task-specific features)")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  - Trainable params: {trainable:,} / {total:,} ({100*trainable//total}%)")
    
    return model


# =====================================================================
# PART 3: TRAINING
# =====================================================================

def train_resnet_transfer(data_dir="data/sampled_subset", epochs=10, batch_size=32):
    """
    Train ResNet-50 with transfer learning.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return
    
    # =====================================================================
    # Data transforms (ResNet expects 224x224, but CIFAKE is 32x32)
    # =====================================================================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ← Resize to ResNet input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # =====================================================================
    # Load dataset
    # =====================================================================
    print(f"\nLoading data from: {data_dir}")
    full_dataset = CIFAKESubset(data_dir, transform=train_transform)
    print(f"Total images: {len(full_dataset)}")
    print(f"  Real: {sum(1 for l in full_dataset.labels if l == 0)}")
    print(f"  Fake: {sum(1 for l in full_dataset.labels if l == 1)}")
    
    # Split: 60% train, 20% val, 20% test
    total = len(full_dataset)
    train_size = int(0.6 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # =====================================================================
    # Create model
    # =====================================================================
    model = create_resnet_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize trainable parameters
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001
    )
    
    # =====================================================================
    # Training loop
    # =====================================================================
    history = {"train_loss": [], "val_acc": [], "val_loss": []}
    
    print("\n" + "="*70)
    print("TRAINING ResNet-50 (Transfer Learning)")
    print("="*70)
    
    for epoch in range(epochs):
        # --------- TRAINING PHASE ---------
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        
        # --------- VALIDATION PHASE ---------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
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
    
    # Track per-class accuracy
    real_correct = fake_correct = 0
    real_total = fake_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Per-class tracking
            for i in range(len(labels)):
                if labels[i] == 0:  # Real
                    real_total += 1
                    if predicted[i] == 0:
                        real_correct += 1
                else:  # Fake
                    fake_total += 1
                    if predicted[i] == 1:
                        fake_correct += 1
    
    test_acc = 100 * test_correct / test_total
    real_acc = 100 * real_correct / real_total if real_total > 0 else 0
    fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
    
    print(f"\n✓ Overall Test Accuracy: {test_acc:.2f}%")
    print(f"  - Real images accuracy: {real_acc:.2f}% ({real_correct}/{real_total})")
    print(f"  - Fake images accuracy: {fake_acc:.2f}% ({fake_correct}/{fake_total})")
    
    # =====================================================================
    # Save model
    # =====================================================================
    model_path = "models/resnet_transfer.pth"
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save metrics
    metrics = {
        "model": "ResNet-50 Transfer Learning",
        "test_accuracy": test_acc,
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "val_accuracy": max(history["val_acc"]),
        "final_train_loss": history["train_loss"][-1],
    }
    with open("models/resnet_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: models/resnet_metrics.json")
    
    # =====================================================================
    # Plot comparison: Baseline vs ResNet
    # =====================================================================
    print("\n" + "="*70)
    print("COMPARISON: Baseline CNN vs ResNet-50")
    print("="*70)
    
    try:
        with open("models/baseline_metrics.json") as f:
            baseline_metrics = json.load(f)
        
        baseline_acc = baseline_metrics.get("test_accuracy", 0)
        improvement = test_acc - baseline_acc
        
        print(f"\nBaseline CNN:        {baseline_acc:.2f}%")
        print(f"ResNet-50:           {test_acc:.2f}%")
        print(f"Improvement:         +{improvement:.2f}% ({100*improvement/baseline_acc:.1f}% relative gain)")
        
        # Plot
        models_list = ["Baseline CNN\n(from scratch)", "ResNet-50\n(Transfer Learning)"]
        accuracies = [baseline_acc, test_acc]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models_list, accuracies, color=['#FF6B6B', '#4ECDC4'], width=0.6, edgecolor='black', linewidth=2)
        plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Baseline CNN vs ResNet-50 Transfer Learning', fontsize=14, fontweight='bold')
        plt.ylim([70, 100])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("models/comparison_baseline_vs_resnet.png", dpi=150)
        print(f"\n✓ Comparison plot saved to: models/comparison_baseline_vs_resnet.png")
        
    except FileNotFoundError:
        print("\n⚠️  Baseline metrics not found. Run train_baseline_cnn.py first.")
    
    # =====================================================================
    # Training curves
    # =====================================================================
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("ResNet-50: Loss over epochs")
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Accuracy", marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("ResNet-50: Validation Accuracy over epochs")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("models/resnet_training_curves.png", dpi=150)
    print(f"✓ Training curves saved to: models/resnet_training_curves.png")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"ResNet-50 Transfer Learning Test Accuracy: {test_acc:.2f}%")
    print(f"\nNext week: Vision Transformer (ViT) comparison + Grad-CAM explainability")


if __name__ == "__main__":
    train_resnet_transfer(
        data_dir="data/sampled_subset",
        epochs=10,
        batch_size=16  # ← Reduced from 32 because ResNet is larger and uses more memory
    )
    
    print("\n✓ Week 2 complete!")
    print("Next: Week 3 - Vision Transformer (ViT)")