# app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import pandas as pd 

# --- 1. Load the Trained Model ---
@st.cache_resource
def load_best_model():
    try:
        model_path = os.path.join("trained_models", "MobileNet_best_model.h5")
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            st.error("Model file not found. Please ensure 'best_model.h5' is in the 'trained_models' directory.")
            return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_best_model()

# --- 2. Define Class Names ---
# This list should match the class names from your training dataset
# in the same order as the directory structure.
class_names = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']


# --- 3. Streamlit UI and Logic ---

# Set up the app title and description
st.set_page_config(
    page_title="Fish Image Classifier",
    page_icon="🐟"
)

st.title("🐟 Fish Image Classifier")
st.write("Upload an image of a fish to get its category and confidence score.")

if model is not None:
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a fish image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Fish Image', use_container_width=True) # <-- Updated parameter
        st.write("")
        st.write("Classifying...")

        # Preprocess the image for the model
        img = image.resize((224, 224))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Normalize pixel values

        # Make a prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(score)
        predicted_class_name = class_names[predicted_class_index]
        confidence = 100 * np.max(score)

        # Display the results
        st.success(f"Prediction: **{predicted_class_name}**")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.write("---")
        st.write("All Class Confidence Scores:")
        # Display a bar chart of all class scores
        prediction_df = pd.DataFrame(score.numpy(), index=class_names, columns=['Confidence'])
        st.bar_chart(prediction_df)
