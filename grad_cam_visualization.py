"""
Grad-CAM Visualization for ResNet-50 Transfer Learning

This script:
1. Loads the ResNet model trained in train_resnet_transfer.py
2. Generates Grad-CAM heatmaps showing which image regions triggered decisions
3. Saves visualizations

Model location: models/resnet_transfer.pth
Data location: data/sampled_subset/ (or your actual data folder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from PIL import Image
import json

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GRAD-CAM FOR RESNET-50 TRANSFER LEARNING                  ║
║                                                                              ║
║  Shows which parts of images made the model decide REAL or FAKE             ║
║  Red regions = high importance for the decision                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =====================================================================
# DATASET LOADER
# =====================================================================

class CIFAKEDataset(Dataset):
    """Load images from data/sampled_subset/ or your data folder"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Try sampled_subset structure
        real_dir = self.data_dir / "real"
        fake_dir = self.data_dir / "fake"
        
        # If not found, try train folder structure
        if not real_dir.exists():
            real_dir = self.data_dir / "train" / "REAL"
        if not fake_dir.exists():
            fake_dir = self.data_dir / "train" / "FAKE"
        
        # Load real images (label 0)
        if real_dir.exists():
            for img_path in sorted(real_dir.glob("*.*")):
                if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    self.images.append(img_path)
                    self.labels.append(0)
        
        # Load fake images (label 1)
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob("*.*")):
                if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    self.images.append(img_path)
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            img = Image.new('RGB', (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, str(img_path)


# =====================================================================
# GRAD-CAM IMPLEMENTATION
# =====================================================================

class GradCAM:
    """
    Compute Grad-CAM for visualizing model decisions.
    
    How it works:
    1. Forward pass: image → model → prediction
    2. Backward pass: compute gradients of prediction w.r.t feature maps
    3. Weight feature maps by gradients
    4. Generate heatmap showing important regions
    """
    
    def __init__(self, model, target_layer_name="layer4"):
        """
        Args:
            model: ResNet-50 model
            target_layer_name: Which layer to visualize from (usually last conv layer)
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self.device = next(model.parameters()).device
        
        # Register hooks to capture activations and gradients
        self._register_hooks(target_layer_name)
    
    def _register_hooks(self, target_layer_name):
        """Attach hooks to capture forward and backward passes"""
        
        def forward_hook(module, input, output):
            """Capture activations during forward pass"""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Capture gradients during backward pass"""
            self.gradients = grad_output[0].detach()
        
        # Find and hook the target layer
        target_layer = dict(self.model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, image_tensor, class_idx):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image_tensor: (3, H, W) normalized image tensor
            class_idx: 0 for Real, 1 for Fake
        
        Returns:
            heatmap: (H, W) numpy array [0, 1]
        """
        # Forward pass
        self.model.eval()
        image_input = image_tensor.unsqueeze(0).to(self.device)  # Add batch dim
        
        with torch.enable_grad():
            output = self.model(image_input)
            class_score = output[0, class_idx]
        
        # Backward pass (compute gradients)
        self.model.zero_grad()
        class_score.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Weight each channel by its gradient
        weights = gradients.mean(dim=(1, 2))  # (C,) - average across spatial dims
        
        # Weighted sum of activations
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=0)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        return cam.cpu().numpy()


# =====================================================================
# VISUALIZATION
# =====================================================================

class Visualizer:
    """Create nice visualizations of Grad-CAM results"""
    
    @staticmethod
    def create_visualization(original_image, heatmap, prediction, confidence, 
                            true_label, output_path=None):
        """
        Create a 3-panel visualization:
        Panel 1: Original image
        Panel 2: Heatmap overlay
        Panel 3: Prediction result
        
        Args:
            original_image: (H, W, 3) numpy array [0, 1]
            heatmap: (H, W) numpy array [0, 1]
            prediction: "REAL" or "FAKE"
            confidence: float [0, 1]
            true_label: "REAL" or "FAKE"
            output_path: Where to save the figure
        """
        
        # Ensure shapes match
        if heatmap.shape != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"True: {true_label} | Predicted: {prediction} ({confidence*100:.1f}%)", 
                     fontsize=14, fontweight="bold")
        
        # Panel 1: Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        
        # Panel 2: Heatmap overlay (Grad-CAM)
        axes[1].imshow(original_image)
        im = axes[1].imshow(heatmap, cmap="jet", alpha=0.6)
        axes[1].set_title("Grad-CAM Heatmap\n(Red = Important for Decision)", 
                         fontsize=12, fontweight="bold")
        axes[1].axis("off")
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Importance", fontsize=10)
        
        # Panel 3: Prediction with color coding
        axes[2].imshow(original_image)
        if prediction == "FAKE":
            color = "red"
            title_text = "🔴 FAKE DETECTED"
        else:
            color = "green"
            title_text = "🟢 REAL IMAGE"
        
        axes[2].set_title(f"{title_text}\nConfidence: {confidence*100:.1f}%", 
                         fontsize=12, fontweight="bold", color=color)
        axes[2].axis("off")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Saved: {output_path}")
        
        plt.close()


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_grad_cam(model_path="models/resnet_transfer.pth", 
                 data_dir="data/sampled_subset",
                 num_samples=10):
    """
    Run Grad-CAM analysis on test images.
    
    Args:
        model_path: Path to saved ResNet model
        data_dir: Path to image data
        num_samples: Number of images to visualize
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}\n")
    
    # =====================================================================
    # 1. LOAD MODEL
    # =====================================================================
    print("Loading ResNet-50 model...")
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"✗ Model not found at: {model_path}")
        print(f"Please train the model first using: python train_resnet_transfer.py")
        return
    
    # Create model architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 2)  # Binary classification
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded model from: {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # =====================================================================
    # 2. LOAD DATA
    # =====================================================================
    print(f"\nLoading data from: {data_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = CIFAKEDataset(data_dir, transform=transform)
    print(f"✓ Found {len(dataset)} images")
    
    if len(dataset) == 0:
        print("✗ No images found! Check your data folder structure.")
        return
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # =====================================================================
    # 3. INITIALIZE GRAD-CAM
    # =====================================================================
    print("\nInitializing Grad-CAM...")
    grad_cam = GradCAM(model, target_layer_name="layer4")
    visualizer = Visualizer()
    
    output_dir = Path("grad_cam_results")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Grad-CAM ready. Will save to: {output_dir}/")
    
    # =====================================================================
    # 4. GENERATE VISUALIZATIONS
    # =====================================================================
    print(f"\nGenerating Grad-CAM visualizations for {num_samples} images...\n")
    
    results_log = []
    correct_count = 0
    
    for idx, (image_tensor, label, image_path) in enumerate(loader):
        if idx >= num_samples:
            break
        
        image_tensor = image_tensor.to(device)
        label_val = label.item()
        
        # Prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probabilities).item()
            confidence = probabilities[pred_class].item()
        
        # Generate Grad-CAM
        heatmap = grad_cam.generate_cam(image_tensor.squeeze(), pred_class)
        
        # Prepare image for visualization (denormalize)
        image_np = image_tensor.squeeze().cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))  # CHW → HWC
        
        # Denormalize using ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np * std + mean).clip(0, 1)
        
        # Labels
        true_label = "REAL" if label_val == 0 else "FAKE"
        pred_label = "REAL" if pred_class == 0 else "FAKE"
        is_correct = (label_val == pred_class)
        
        if is_correct:
            correct_count += 1
        
        print(f"Image {idx+1}/{num_samples}:")
        print(f"  True: {true_label}, Predicted: {pred_label} ({confidence*100:.1f}%)")
        print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        
        # Save visualization
        output_path = output_dir / f"sample_{idx:02d}_{pred_label}_{confidence*100:.0f}pct.png"
        visualizer.create_visualization(
            image_np, heatmap, pred_label, confidence, true_label, 
            output_path=output_path
        )
        
        # Log result
        results_log.append({
            "image_idx": idx,
            "image_path": str(image_path),
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": float(confidence),
            "is_correct": is_correct,
            "visualization": str(output_path)
        })
    
    # =====================================================================
    # 5. SAVE RESULTS
    # =====================================================================
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_log, f, indent=2)
    
    accuracy = 100 * correct_count / len(results_log)
    
    print("\n" + "="*70)
    print("GRAD-CAM ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✓ Visualizations saved to: {output_dir}/")
    print(f"\nResults:")
    print(f"  Images analyzed: {len(results_log)}")
    print(f"  Correct predictions: {correct_count}/{len(results_log)}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"\n✓ Results log saved to: {output_dir}/results.json")
    print(f"\nWhat the visualizations show:")
    print(f"  - Left panel: Original image")
    print(f"  - Middle panel: Red regions = important for decision")
    print(f"  - Right panel: Model's prediction & confidence")


if __name__ == "__main__":
    run_grad_cam(
        model_path="models/resnet_transfer.pth",
        data_dir="data/sampled_subset",  # Change if your data is elsewhere
        num_samples=10
    )
    
    print("\n" + "="*70)
    print("✓ Week 4-5: Grad-CAM Explainability Complete!")
    print("✓ Next: Week 6 - Streamlit Interactive Demo")
    print("="*70)