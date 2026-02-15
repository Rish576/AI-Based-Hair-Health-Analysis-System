import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="AI Hair Health Analysis",
    page_icon="💇‍♀️",
    layout="centered"
)

st.title("💇‍♀️ AI Hair Health Analysis")
st.write("Upload your scalp image to check hair health.")

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("hair_model.h5")
    return model

model = load_model()

classes = ["Dandruff", "Hair Loss", "Healthy", "Oily"]

# ----------------------------------
# Suggestions in Languages
# ----------------------------------
suggestions = {
    "Hair Loss": {
        "English": "Eat protein food, use mild shampoo, consult doctor.",
        "Telugu": "ప్రోటీన్ ఉన్న ఆహారం తినండి, తేలికపాటి షాంపూ వాడండి.",
        "Hindi": "प्रोटीन वाला खाना खाएं, हल्का शैम्पू इस्तेमाल करें।"
    },
    "Dandruff": {
        "English": "Use anti-dandruff shampoo twice weekly.",
        "Telugu": "డాండ్రఫ్ షాంపూ వారానికి 2 సార్లు వాడండి.",
        "Hindi": "एंटी डैंड्रफ शैम्पू हफ्ते में 2 बार लगाएं।"
    },
    "Oily": {
        "English": "Wash hair regularly and avoid heavy oil.",
        "Telugu": "తలస్నానం తరచుగా చేయండి.",
        "Hindi": "बाल नियमित धोएं।"
    },
    "Healthy": {
        "English": "Your scalp looks healthy!",
        "Telugu": "మీ తల ఆరోగ్యంగా ఉంది!",
        "Hindi": "आपका स्कैल्प स्वस्थ है!"
    }
}

# ----------------------------------
# Language Selection
# ----------------------------------
lang = st.selectbox("Choose Language", ["English", "Telugu", "Hindi"])

# ----------------------------------
# Image Upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload Scalp Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------
# Prediction
# ----------------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file).resize((128,128))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    st.success(f"Prediction: **{result}**")
    st.info(suggestions[result][lang])
