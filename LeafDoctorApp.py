import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import google.generativeai as genai
import os
import uuid


# Configure Google Generative AI
genai.configure(api_key="AIzaSyAZ9x6ig-GikTC8Lf0hYFkBH_zjCZcBT64")

# Load your trained model
model = tf.keras.models.load_model('trained_model.keras')

# Class names for the plant diseases
class_names = ['Apple - Apple Scab', 
 'Apple - Black Rot', 
 'Apple - Cedar Apple Rust', 
 'Apple - Healthy', 
 'Blueberry - Healthy', 
 'Cherry (including sour) - Powdery Mildew', 
 'Cherry (including sour) - Healthy', 
 'Corn (maize) - Cercospora Leaf Spot Gray Leaf Spot', 
 'Corn (maize) - Common Rust', 
 'Corn (maize) - Northern Leaf Blight', 
 'Corn (maize) - Healthy', 
 'Grape - Black Rot', 
 'Grape - Esca (Black Measles)', 
 'Grape - Leaf Blight (Isariopsis Leaf Spot)', 
 'Grape - Healthy', 
 'Orange - Haunglongbing (Citrus Greening)', 
 'Peach - Bacterial Spot', 
 'Peach - Healthy', 
 'Pepper (bell) - Bacterial Spot', 
 'Pepper (bell) - Healthy', 
 'Potato - Early Blight', 
 'Potato - Late Blight', 
 'Potato - Healthy', 
 'Raspberry - Healthy', 
 'Soybean - Healthy', 
 'Squash - Powdery Mildew', 
 'Strawberry - Leaf Scorch', 
 'Strawberry - Healthy', 
 'Tomato - Bacterial Spot', 
 'Tomato - Early Blight', 
 'Tomato - Late Blight', 
 'Tomato - Leaf Mold', 
 'Tomato - Septoria Leaf Spot', 
 'Tomato - Spider Mites (Two-spotted Spider Mite)', 
 'Tomato - Target Spot', 
 'Tomato - Tomato Yellow Leaf Curl Virus', 
 'Tomato - Tomato Mosaic Virus', 
 'Tomato - Healthy']

# Streamlit app title and description
st.set_page_config(page_title="Leaf Doctor", page_icon="assets/NavLogo.png", layout="centered")


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.pinimg.com/564x/61/33/1b/61331bf9d77178110266d9284a74ec35.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    """, unsafe_allow_html=True
)

col1, col2 = st.columns([1, 10])  # Create two columns, adjust ratios as needed

with col1:
    st.image("assets/NavLogo.png", width=55)  # Adjust width as per the image size

with col2:
    st.title("Leaf Doctor")
st.write("Upload an image of a plant leaf, and this app will predict the disease and suggest preventive measures and treatments.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def fetch_prevention_and_treatments(disease_name):
    """Fetch prevention and treatments from Google Gemini AI."""
    prompt = f"What are the prevention and treatment methods for {disease_name} in plants?"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

def predict_disease(image):
    
    # Create a folder to save the image if it doesn't exist
    folder = "uploaded_images"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Generate a unique filename for the image to avoid conflicts
    image_name = f"leaf_{uuid.uuid4()}.png"
    image_path = os.path.join(folder, image_name)
    
    # Save the uploaded PIL image to the folder
    image.save(image_path)

    img = cv2.imread(image_path)
    # Convert the image to RGB format (OpenCV loads images in BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size = (128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])   

    # Make the prediction using the loaded model
    prediction = model.predict(input_arr)

    # Get the index of the predicted class and map it to the class name
    result_index = np.argmax(prediction)
    predicted_class = class_names[result_index]
    
    # Delete image from folder
    if os.path.exists(image_path):
        os.remove(image_path)
    return predicted_class


if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Predict the disease when the user clicks the button
    if st.button("Predict Disease"):
        with st.spinner("Analyzing the image..."):
            predicted_disease = predict_disease(image)

        # Display the prediction
        st.success(f"Predicted Disease: **{predicted_disease}**")

        # Fetch prevention and treatment information from Google Generative AI
        with st.spinner("Fetching prevention and treatments..."):
            treatments = fetch_prevention_and_treatments(predicted_disease)

        # Display the prevention and treatment info
        st.subheader("Prevention and Treatments")
        st.write(treatments)
else:
    st.write("Please upload a leaf image to begin.")

# Footer with some styling
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 0px;
    }
    </style>
    <div class="footer">
        <p>© 2024 Leaf Doctor | Powered by TensorFlow & Google Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
