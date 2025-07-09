import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Plant Disease Prediction", "Download Test Images"])

# Home Page
if app_mode == "Home":
    st.header("Plant Disease Recognition System")
    image_path = "home_page.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    image_path = "home_page1.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    st.markdown("""
    #### About Dataset
    The dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.

    A new directory containing 24 test images is created later for prediction purpose.

    #### Content
    1. Train (70,295 images)
    2. Test (24 images)
    3. Valid (17,572 images)
    """)

# Prediction Page
elif app_mode == "Plant Disease Prediction":
    st.header("Plant Disease Prediction")
    test_image = st.file_uploader("Choose an image")
    if st.button("Show Image") and test_image:
        st.image(test_image)
    if st.button("Predict") and test_image:
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        st.success("Model is predicting it's a **{}**".format(class_name[result_index]))

# Download Test Images Page
elif app_mode == "Download Test Images":
    st.header("Download Sample Test Images")

    test_images_folder = "C:\\Users\\khushbu\\OneDrive\\Desktop\\Plant_Disease\\New Plant Diseases Dataset(Augmented)\\test"
    if not os.path.exists(test_images_folder):
        st.error("Test images folder not found. Please check the path.")
    else:
        image_files = [f for f in os.listdir(test_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()

        cols = st.columns(3)  # Display 3 images per row
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(test_images_folder, image_file)
            with cols[idx % 3]:
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=image_file)
                    with open(image_path, "rb") as file:
                        st.download_button(
                            label="Download",
                            data=file,
                            file_name=image_file,
                            mime="image/jpeg"
                        )
                except Exception as e:
                    st.warning(f"Could not load image: {image_file}")
