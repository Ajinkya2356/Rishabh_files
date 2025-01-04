import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Path to your saved model
MODEL_PATH = "./Custom Model/custom.h5"

# Load the pre-trained model
model = load_model(MODEL_PATH)

# Define image size expected by the model
IMG_HEIGHT, IMG_WIDTH = 85, 226

# Function to preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocess the image to match the model's input shape.
    Args:
        image: PIL.Image object
    Returns:
        Preprocessed image ready for prediction
    """
    # Convert the image to grayscale (if required for your model)
    image = image.convert("L")  # 'L' mode for grayscale
    # Resize the image
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    # Convert to numpy array
    img_array = img_to_array(image) / 255.0  # Normalize
    # Add batch dimension and channel dimension
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

# Function to make predictions
def predict_image(image):
    """
    Predict the class of the image using the loaded model.
    Args:
        image: Preprocessed image array
    Returns:
        Prediction result (Defective or Non-Defective)
    """
    prediction = np.round(model.predict(image)[0][0]).astype(int) 
    return "Class 1" if prediction == 1 else "Class 0"
    

# Streamlit App Layout
st.title("Image Classification App")
st.write("Upload an image to classify it as Defective or Non-Defective.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=85)
    st.write("Classifying...")

    # Preprocess the image
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)

    # Predict the result
    prediction = predict_image(preprocessed_image)

    # Display the prediction result
    st.write(f"Prediction: **{prediction}**")
