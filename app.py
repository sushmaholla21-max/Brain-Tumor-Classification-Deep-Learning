import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Set up the page title and look
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classification System")
st.write("Upload an MRI scan to classify it into one of four categories: Glioma, Meningioma, Pituitary, or Normal.")

# 2. Load your trained model (Caching keeps it fast!)
@st.cache_resource
def load_my_model():
    # Replace 'best_model.keras' with the exact filename of your model file if it is different
    return tf.keras.models.load_model('best_model.keras')

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure your model file is uploaded to GitHub.")

# 3. Create the Image Uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
    
    st.write("🔄 Classifying...")
    
    # 4. Preprocess the image to match your model's expected input size
    # (Change 224, 224 if your ResNet/VGG expected a different size like 150x150)
    img_size = (224, 224) 
    img = image.resize(img_size)
    img_array = np.array(img)
    
    # Ensure image has 3 channels (RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:,:,:3]
        
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Make the Prediction
    try:
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        # Define categories matching your dataset classes
        classes = ['Glioma', 'Meningioma', 'Normal (No Tumor)', 'Pituitary']
        predicted_class = classes[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        # Show Results
        st.subheader(f"Prediction: **{predicted_class}**")
        st.progress(int(confidence))
        st.write(f"Confidence Level: **{confidence:.2f}%**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
