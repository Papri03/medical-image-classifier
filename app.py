import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Use Streamlit's cache to load the model only once
@st.cache_resource
def load_model():
    # This function loads your saved model
    # Make sure 'cifar10_model.h5' is in the same directory as this script
    model = tf.keras.models.load_model('cifar10_model.h5')
    return model

# Load the model and class names at the start of the app
model = load_model()

# Define the class names for the dataset the model was trained on
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'] 

# 1. Design the web app interface
st.title('Image Classification Demo')
st.write('Upload an image to get a prediction from a pre-trained model.')

# 2. File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 3. Process the uploaded image and make a prediction
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    # The model was trained on 32x32 images
    image_resized = image.resize((256, 256))
    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = img_array / 255.0 # Normalize pixel values

    # Make the prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(score)
    predicted_class_name = class_names[predicted_class_index]
    confidence = 100 * np.max(score)

    # 4. Display the results
    st.success(f"The uploaded image most likely belongs to **{predicted_class_name}** with a {confidence:.2f}% confidence.")