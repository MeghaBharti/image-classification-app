import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Function to prepare the image and make predictions
def predict_image(img):
    # Resize and preprocess the image to fit ResNet50 input shape
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Decode predictions into readable class names
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Streamlit UI to upload and display images
st.title("Image Classification App with Streamlit")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image using PIL
    img = Image.open(uploaded_file)
    
    # Display the image
    st.image(img, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    
    # Predict the class
    predictions = predict_image(img)
    
    # Display predictions
    st.write("Predictions:")
    for pred in predictions:
        st.write(f"{pred[1]}: {pred[2]*100:.2f}%")
