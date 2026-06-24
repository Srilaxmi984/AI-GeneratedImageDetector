"""
DIAGNOSTIC SCRIPT - Run this FIRST before changing anything else.

This checks:
1. Is the model actually loading correct weights?
2. Is it predicting the SAME class for everything (broken) or just biased (fixable)?
3. What's the actual probability distribution across many images?
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# =====================================================================
# 1. LOAD MODEL AND CHECK ARCHITECTURE MATCH
# =====================================================================
print("="*70)
print("STEP 1: Checking model loading")
print("="*70)

model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, 2)

model_path = Path("models/resnet_transfer.pth")
print(f"Loading from: {model_path}")

checkpoint = torch.load(model_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    print("Checkpoint is a dict with 'model_state_dict' key")
    state_dict = checkpoint["model_state_dict"]
else:
    print("Checkpoint is a raw state_dict")
    state_dict = checkpoint

# Try loading and catch mismatches
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"⚠️  MISSING KEYS (not loaded): {missing[:5]}... ({len(missing)} total)")
if unexpected:
    print(f"⚠️  UNEXPECTED KEYS (ignored): {unexpected[:5]}... ({len(unexpected)} total)")
if not missing and not unexpected:
    print("✓ All weights loaded cleanly, architecture matches")

model = model.to(device)
model.eval()

# =====================================================================
# 2. TEST ON KNOWN REAL AND KNOWN FAKE IMAGES SEPARATELY
# =====================================================================
print("\n" + "="*70)
print("STEP 2: Testing on known REAL images vs known FAKE images")
print("="*70)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def test_folder(folder_path, true_label_name, max_images=20):
    """Test all images in a folder, print prediction distribution"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"  ✗ Folder not found: {folder_path}")
        return None
    
    images = list(folder.glob("*.*"))[:max_images]
    images = [p for p in images if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not images:
        print(f"  ✗ No images found in {folder_path}")
        return None
    
    predictions = []
    confidences = []
    
    for img_path in images:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred = torch.argmax(probs).item()
                conf = probs[pred].item()
            
            predictions.append(pred)
            confidences.append(conf)
        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")
    
    predictions = np.array(predictions)
    real_count = (predictions == 0).sum()
    fake_count = (predictions == 1).sum()
    
    print(f"\n  Folder: {folder_path} (true label: {true_label_name})")
    print(f"  Images tested: {len(predictions)}")
    print(f"  Predicted REAL: {real_count} ({100*real_count/len(predictions):.1f}%)")
    print(f"  Predicted FAKE: {fake_count} ({100*fake_count/len(predictions):.1f}%)")
    print(f"  Avg confidence: {np.mean(confidences)*100:.1f}%")
    print(f"  Confidence range: {np.min(confidences)*100:.1f}% - {np.max(confidences)*100:.1f}%")
    
    return predictions, confidences

# Find your actual data folders
possible_real_paths = [
    "data/sampled_subset/real",
    "data/train/REAL",
    "data/test/REAL"
]
possible_fake_paths = [
    "data/sampled_subset/fake",
    "data/train/FAKE",
    "data/test/FAKE"
]

real_path = None
fake_path = None
for p in possible_real_paths:
    if Path(p).exists():
        real_path = p
        break
for p in possible_fake_paths:
    if Path(p).exists():
        fake_path = p
        break

if real_path:
    test_folder(real_path, "REAL")
else:
    print("✗ Could not find a REAL images folder")

if fake_path:
    test_folder(fake_path, "FAKE")
else:
    print("✗ Could not find a FAKE images folder")

# =====================================================================
# 3. DIAGNOSIS
# =====================================================================
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("""
Look at the results above:

CASE A - Model is BROKEN (always predicts same class regardless of input):
  → Both REAL and FAKE folders show ~100% predicted as the SAME class
  → This means the model weights are wrong, or preprocessing mismatch
  → FIX: Retrain, or check the model loading code

CASE B - Model is BIASED but working (different % for different folders):
  → REAL folder shows mostly REAL predictions, FAKE folder shows mostly FAKE
  → But maybe not as clean as your test accuracy suggested
  → FIX: This could be a threshold/calibration issue - fixable

CASE C - Model works fine on these folders but fails on NEW uploaded images:
  → This means train/test images differ from real-world uploaded images
  → Likely cause: your training images are 32x32 CIFAKE images resized up,
    but real-world photos you upload in Streamlit have very different 
    characteristics (resolution, compression, content)
  → FIX: This is a DATASET MISMATCH problem, not a code bug
""")