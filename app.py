import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import os
import gdown
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_gradcam(model, input_tensor, original_image):
    # 1. Hook setup to capture gradients and feature maps
    feature_maps = []
    gradients = []
    
    def forward_hook(module, input, output):
        feature_maps.append(output)
        
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    # In PyTorch ResNet101, 'layer4' is the final convolutional block
    target_layer = model.layer4[-1]
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    # 2. Forward pass
    model.eval()
    output = model(input_tensor)
    
    # Get the index of the highest predicted class
    idx = output.argmax(dim=1).item()
    
    # 3. Backward pass to calculate gradients
    model.zero_grad()
    output[0, idx].backward()
    
    # Remove the hooks to keep memory clean
    handle_forward.remove()
    handle_backward.remove()
    
    # 4. Compute Grad-CAM weights
    grads = gradients[0].cpu().data.numpy()[0]
    f_maps = feature_maps[0].cpu().data.numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))  # Global average pooling of gradients
    
    # Target feature map weighted combination
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * f_maps[i]
        
    # Apply ReLU to only keep features that positively contribute to the class
    cam = np.maximum(cam, 0)
    
    # Resize map to fit original MRI dimensions
    cam = cv2.resize(cam, (original_image.width, original_image.height))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize
    
    # 5. Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert original PIL image to numpy array
    img_np = np.array(original_image.convert('RGB'))
    
    # Superimpose the heatmap onto original image (alpha blending)
    overlaid_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    return overlaid_img

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classification System")
st.write("Upload an MRI scan to classify it into one of four categories.")

# 1. Download model from Google Drive if it doesn't exist locally
MODEL_PATH = "best_model.pth"

@st.cache_resource
def load_pytorch_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive... This might take a minute."):
            # PASTE YOUR GOOGLE DRIVE
            url = "https://drive.google.com/file/d/1HPANEALJFQOUlHgrWGMc09FlJjSa8dtG/view?usp=sharing"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    # Assuming standard ResNet/VGG setup. Adjust structure if you custom built it
    # We load it directly onto the CPU for the server
    model = models.resnet101(weights=None) # Change to models.vgg19 if you used VGG19
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 4)
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_pytorch_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. Image Uploader & Preprocessing
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
    
    # Standard deep learning transforms (match your training setup)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)
    
    st.write("🔄 Classifying...")
    
    with torch.no_grad():
        outputs = model(batch_t)
        _, preds = torch.max(outputs, 1)
        
        classes = ['Glioma', 'Meningioma', 'Normal (No Tumor)', 'Pituitary']
        predicted_class = classes[preds[0].item()]
        
        st.subheader(f"Prediction: **{predicted_class}**")
        # Generate the heatmap
        # (Assuming 'image_tensor' is your preprocessed tensor and 'image' is the loaded PIL image)
        with st.spinner("Generating Grad-CAM Heatmap..."):
            heatmap_result = generate_gradcam(model, image_tensor, image)
            
        # Display side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original MRI Scan", use_column_width=True)
        with col2:
            st.image(heatmap_result, caption="Grad-CAM Tumor Localization Map (Red indicates high focus)", use_column_width=True)
