import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Use Streamlit's cache to load the model only once
@st.cache_resource
def load_model():
    # This function loads your saved model
    # Make sure your brain tumor model file is named 'cifar10_model.h5'
    model = tf.keras.models.load_model('cifar10_model.h5')
    return model

# Load the model and class names at the start of the app
model = load_model()

# Define the class names for your brain tumor model
# Adjust these to match the classes your model was trained on
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'] # Example classes

# 1. Design the web app interface
st.title('Brain Tumor Classification App')
st.write('Upload a brain MRI image to get a tumor classification prediction.')

# 2. File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 3. Process the uploaded image and make a prediction
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # Preprocess the image for the model
        # **IMPORTANT: REPLACE (150, 150) with your model's required input size**
        # For example, (256, 256), (224, 224), etc.
        image_resized = image.resize((150, 150))
        
        # Convert image to a numpy array and ensure it's a float type
        img_array = np.array(image_resized).astype('float32')

        # Add a batch dimension at the beginning (required by Keras)
        img_array = np.expand_dims(img_array, axis=0) 

        # Normalize the pixel values (assuming model was trained on normalized data)
        # This is the most common normalization for image models
        img_array = img_array / 255.0

        # Make the prediction
        predictions = model.predict(img_array)
        
        # Get the predicted class and confidence
        score = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(score)
        predicted_class_name = class_names[predicted_class_index]
        confidence = 100 * np.max(score)

        # 4. Display the results
        st.success(f"The uploaded image most likely belongs to **{predicted_class_name}** with a {confidence:.2f}% confidence.")
        
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        st.warning("Please ensure the uploaded image is a valid brain MRI and that the model's expected input shape is correctly set in the code.")