"""
Week 6: Streamlit Interactive App
AI-Generated Image Detector with Grad-CAM Visualization

Run with:
  streamlit run streamlit_app.py

This creates an interactive web app where users can:
1. Upload an image
2. Get prediction (REAL or FAKE)
3. See Grad-CAM heatmap showing why
4. View confidence scores
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# =====================================================================
# PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .fake-box {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid red;
    }
    .real-box {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid green;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# GRAD-CAM CLASS
# =====================================================================

class GradCAM:
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.gradients = None
        self.activations = None
        self.device = next(model.parameters()).device
        self._register_hooks(target_layer_name)
    
    def _register_hooks(self, target_layer_name):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer = dict(self.model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, image_tensor, class_idx):
        self.model.eval()
        image_input = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.enable_grad():
            output = self.model(image_input)
            class_score = output[0, class_idx]
        
        self.model.zero_grad()
        class_score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2))
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=0)
        cam = F.relu(cam)
        
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        return cam.cpu().numpy()


# =====================================================================
# LOAD MODEL (with caching for speed)
# =====================================================================

@st.cache_resource
def load_model():
    """Load ResNet-50 model (cached so it only loads once)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 2)
    
    model_path = Path("models/resnet_transfer.pth")
    
    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        st.info("Please train the model first: `python train_resnet_transfer.py`")
        st.stop()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    
    model = model.to(device)
    return model, device


# =====================================================================
# PREDICTION FUNCTION
# =====================================================================

def predict_and_explain(image_pil, model, device):
    """
    Get prediction and Grad-CAM visualization.
    
    Returns:
        prediction: "REAL" or "FAKE"
        confidence: float [0, 1]
        heatmap: numpy array for visualization
        original_image_np: denormalized image
    """
    
    # IMPORTANT: The model was trained on CIFAKE, which is 32x32 pixels.
    # Real-world uploaded photos are much higher resolution with different
    # detail/compression characteristics. To reduce this domain gap, we
    # first downscale to 32x32 (matching training distribution) before
    # upscaling to 224x224 for the model. This is a partial mitigation,
    # not a full fix -- the model still wasn't trained on true high-res
    # real-vs-fake features.
    image_pil_small = image_pil.resize((32, 32), Image.BICUBIC)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image_pil_small)
    image_input = image_tensor.unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        output = model(image_input)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probabilities).item()
        confidence = probabilities[pred_class].item()
    
    # Grad-CAM
    grad_cam = GradCAM(model, target_layer_name="layer4")
    heatmap = grad_cam.generate_cam(image_tensor, pred_class)
    
    # Denormalize image
    image_np = image_tensor.cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np * std + mean).clip(0, 1)
    
    # Upsample heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    
    prediction = "REAL" if pred_class == 0 else "FAKE"
    
    return prediction, confidence, heatmap, image_np


# =====================================================================
# STREAMLIT APP
# =====================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🔍 AI-Generated Image Detector</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="subheader">Detect fake images using ResNet-50 + Grad-CAM explainability</div>', 
                unsafe_allow_html=True)
    
    # Sidebar info
    with st.sidebar:
        st.title("About This Project")
        st.markdown("""
        **AI Image Detector** identifies whether an image is:
        - 🟢 **Real**: Authentic photograph
        - 🔴 **Fake**: AI-generated (DALL-E, Midjourney, Stable Diffusion, etc.)
        
        **Technologies Used:**
        - ResNet-50 (Transfer Learning)
        - Grad-CAM (Explainability)
        - Streamlit (Web Interface)
        
        **Accuracy:** 91.98% on test set
        """)
        
        st.divider()
        
        st.markdown("**How it works:**")
        st.markdown("""
        1. Upload an image (JPG, PNG)
        2. Model makes prediction
        3. Grad-CAM shows which parts triggered the decision
        4. Red regions = high importance
        """)
        
        st.divider()
        
        st.markdown("**⚠️ Known Limitation:**")
        st.caption("""
        This model is trained on CIFAKE, a benchmark dataset of 
        32×32 pixel images. Real-world photos (e.g. from a phone 
        camera) have very different detail and compression 
        characteristics after being resized, so predictions on 
        such images may be less reliable than on the original 
        benchmark. A production system would need retraining on 
        higher-resolution, real-world images to close this gap.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a JPG or PNG image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True, caption="Uploaded Image")
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔬 Analysis")
            
            # Load model
            with st.spinner("Loading model..."):
                model, device = load_model()
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                prediction, confidence, heatmap, image_np = predict_and_explain(
                    image, model, device
                )
            
            # Display prediction
            if prediction == "FAKE":
                st.markdown(
                    f'<div class="fake-box">'
                    f'<h2 style="color: red; margin: 0;">🔴 LIKELY FAKE</h2>'
                    f'<p style="margin: 10px 0;">Confidence: <strong>{confidence*100:.1f}%</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="real-box">'
                    f'<h2 style="color: green; margin: 0;">🟢 LIKELY REAL</h2>'
                    f'<p style="margin: 10px 0;">Confidence: <strong>{confidence*100:.1f}%</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            st.divider()
            
            # Visualizations
            st.subheader("🎨 Grad-CAM Visualization")
            st.markdown("""
            The heatmap below shows which regions of the image influenced the model's decision:
            - **Red/Hot areas** = High importance for the prediction
            - **Blue/Cool areas** = Low importance
            """)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original
            axes[0].imshow(image_np)
            axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
            axes[0].axis("off")
            
            # Heatmap overlay
            axes[1].imshow(image_np)
            im = axes[1].imshow(heatmap, cmap="jet", alpha=0.6)
            axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1])
            
            # Prediction
            axes[2].imshow(image_np)
            color = "red" if prediction == "FAKE" else "green"
            axes[2].set_title(
                f"{prediction}\n{confidence*100:.1f}% confidence",
                fontsize=12, fontweight="bold", color=color
            )
            axes[2].axis("off")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.divider()
            
            # Interpretation
            st.subheader("📊 What This Means")
            
            if prediction == "FAKE":
                st.warning("""
                **This image appears to be AI-generated.**
                
                Common artifacts in AI-generated images:
                - Unnatural textures and patterns
                - Inconsistent lighting
                - Strange hand/finger structures
                - Blurred or distorted regions
                - Symmetric patterns
                
                **Note:** The model is 91.98% accurate but not perfect. 
                Always verify with other methods if critical.
                """)
            else:
                st.success("""
                **This image appears to be authentic.**
                
                Real images typically show:
                - Natural textures and lighting
                - Realistic details and imperfections
                - Consistent spatial relationships
                - Natural variations
                
                **Note:** Some very convincing AI images may still pass. 
                Use multiple detection methods for critical applications.
                """)
        
        else:
            st.info("👈 Upload an image to get started!")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "ResNet-50")
    with col2:
        st.metric("Accuracy", "91.98%")
    with col3:
        st.metric("Dataset", "CIFAKE")


if __name__ == "__main__":
    main()