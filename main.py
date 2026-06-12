import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.image(r"C:\Users\Gopal Sharma\Desktop\project 2\Fake Face Detection\Testing\Home page.png",width=4, use_container_width=True)

with col2:
    st.markdown("## Fake Face Detection System 🧠🕵️‍♂️")
    st.write(
        "Upload or capture a face image to check whether it is **Real or Fake** "
        "using a Deep Learning CNN model."
    )

# ---------------- BUTTON STYLE ----------------
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #4CAF50; /* Change button color */
            color: white; /* Text color */
            border: 2px solid white; /* White border */
            font-size: 15px;
            padding: 5px 15px;
            border-radius: 5px;
            transition: 0.3s;

        }
        .stButton>button:hover {
            background-color: blue;  /* Change color to blue on hover */
            color: white;
            border: 2px solid white;
            
        .stButton>button:active {
        background-color: red; /* Change color after clicking */
        color: white;
        border: 2px solid white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- MODEL PREDICTION FUNCTION ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Users\Gopal Sharma\Desktop\project 2\Fake Face Detection\Trained_model.keras")

model = load_model()

def model_prediction(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    confidence = np.max(prediction) * 100
    class_index = np.argmax(prediction)

    return class_index, confidence


# ---------------- SIDEBAR ----------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Fake Face Detection"]
)

# ---------------- HOME ----------------
if app_mode == "Home":
    st.markdown(
        """
        ### How It Works:
        1. **Upload or Capture Image:** Use gallery upload or camera input.
        2. **Face Analysis:** CNN model analyzes facial features.
        3. **Prediction:** Classifies the face as **Real or Fake** with confidence.

        ### Why This System?
        - AI-based Deepfake Detection  
        - Works on human face images only  
        - Fast, automated & reliable  

        👉 Go to **Fake Face Detection** to try it now.
        """
    )

# ---------------- ABOUT ----------------
elif app_mode == "About":
    st.header("About the Project")
    st.markdown(
        """
        #### About Dataset
        This dataset consists of about 10K rgb images of Real And Fake Faces images. which is categorized into 2 different classes.The total dataset is divided into 70/30 ratio of training and validation set preserving the directory structure.
        
        #### Fake Face Detection using Deep Learning
        This project detects whether a human face image is **Real or AI-generated (Fake)**.
        The system is trained using CNN on datasets like:

        - RVF10K (Real vs Fake Faces)

        #### Key Features:
        - CNN-based classification
        - Image preprocessing & normalization
        - Confidence score generation
        - Supports gallery & camera images
        """
    )

# ---------------- PREDICTION ----------------
elif app_mode == "Fake Face Detection":
    st.header("Fake Face Detection")

    option = st.radio(
        "Select Image Input Method:",
        ["Upload from Gallery", "Capture from Camera"]
    )

    image = None

    # -------- GALLERY UPLOAD --------
    if option == "Upload from Gallery":
        uploaded_file = st.file_uploader(
            "Upload a face image", type=["jpg", "png", "jpeg"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------- CAMERA INPUT --------
    elif option == "Capture from Camera":
        camera_image = st.camera_input("Capture a face image")
        if camera_image:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Image", use_container_width=True)

    # -------- PREDICT --------
    if image is not None and st.button("Predict"):
        with st.spinner("Analyzing face..."):
            st.snow()
            st.write("Our Prediction")
            class_name, confidence = model_prediction(image)

        class_names = ["Fake", "Real"]

        st.success(
            f"Prediction: **{class_names[class_name]}**\n\n"
            f"Confidence: **{confidence:.2f}%**"
        )
