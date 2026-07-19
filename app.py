import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import os
import gdown

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
        nn.Linear(num_ftrs, 512)
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4)
    
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
