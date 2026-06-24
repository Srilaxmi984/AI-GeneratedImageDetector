import numpy as np
from pathlib import Path
import shutil

SUBSET_DIR = Path("data/sampled_subset")
SUBSET_DIR.mkdir(parents=True, exist_ok=True)

def sample_cifake_subset(cifake_root, sample_size=2000):

    real_train = cifake_root / "train/REAL"
    fake_train = cifake_root / "train/FAKE"

    real_test = cifake_root / "test/REAL"
    fake_test = cifake_root / "test/FAKE"

    # ✅ FIX: support jpg images
    real_images = list(real_train.rglob("*.jpg")) + list(real_test.rglob("*.jpg"))
    fake_images = list(fake_train.rglob("*.jpg")) + list(fake_test.rglob("*.jpg"))

    print(f"\nFound {len(real_images)} real images, {len(fake_images)} fake images")

    if len(real_images) == 0 or len(fake_images) == 0:
        print("❌ No images found!")
        return

    np.random.seed(42)

    sampled_real = np.random.choice(real_images, size=2000, replace=False)
    sampled_fake = np.random.choice(fake_images, size=2000, replace=False)

    print(f"Sampled {len(sampled_real)} real + {len(sampled_fake)} fake")

    real_subset = SUBSET_DIR / "real"
    fake_subset = SUBSET_DIR / "fake"

    real_subset.mkdir(parents=True, exist_ok=True)
    fake_subset.mkdir(parents=True, exist_ok=True)

    for img in sampled_real:
        shutil.copy(img, real_subset / img.name)

    for img in sampled_fake:
        shutil.copy(img, fake_subset / img.name)

    print("✅ Done! data/sampled_subset ready")


print("Starting sampling...")

cifake_path = Path("data/cifake")

if cifake_path.exists():
    sample_cifake_subset(cifake_path)
else:
    print("❌ CIFAKE folder not found")