
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import gdown
import os
from PIL import Image

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬"
)

st.title("Skin Cancer Detection")
st.write("Upload a skin image to see the model's predicted class.")

MODEL_PATH = "skin_cancer_model.keras"
FILE_ID = "1wL2RDTaVE4HQ3KKVgtKtCumIBio27uw2"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

@st.cache_data
def load_classes():
    with open("class_names.json", "r", encoding="utf-8") as f:
        return json.load(f)

try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error("Could not load model or class names.")
    st.exception(e)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        resized_image = image.resize((224, 224))
        image_array = np.array(resized_image, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        try:
            predictions = model(image_array, training=False).numpy()[0]
            predicted_index = int(np.argmax(predictions))

            if predicted_index >= len(class_names):
                st.error("Model classes do not match class names.")
                st.stop()

            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            st.subheader("Prediction")
            st.write(f"Predicted class: **{predicted_class}**")
            st.write(f"Model confidence: **{confidence:.2f}%**")

            st.warning(
                "Educational prediction only. This is not a medical diagnosis. "
                "Please consult a qualified dermatologist."
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


          
