import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import numpy as np

# Load your model weights
@st.cache_resource
def load_my_model():
    model = keras_load_model('best_model.weights.h5')  # Assuming the model weights are saved here
    return model

model = load_my_model()

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size as per your model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)
    return "Right" if predicted_class[0] == 1 else "Wrong"

# Streamlit app design
st.set_page_config(page_title="Image Quality Checker", page_icon="📸", layout="wide")

# App title and header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Image Quality Checker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload an image to determine if it meets quality standards</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Sidebar design
st.sidebar.markdown("### Instructions")
st.sidebar.write("""
1. Upload an image using the 'Upload an image' button.
2. Wait for the result to display below the upload section.
3. Check if the image is 'Right' or 'Wrong.'
""")

# Main app
if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict and display result
    with st.spinner("Analyzing..."):
        result = predict(image)
    
    if result == "Right":
        st.success("✅ The image is **Right** and meets quality standards!")
    else:
        st.error("❌ The image is **Wrong** and does not meet quality standards.")
else:
    st.markdown("<p style='text-align: center; color: red;'>No image uploaded yet. Please upload an image.</p>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: small; color: gray;'>Powered by AI • Built with Streamlit</p>",
    unsafe_allow_html=True
)