
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬"
)

st.title("Skin Cancer Detection")
st.write("Upload a skin image to see the model's predicted class.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "skin_cancer_model.keras",
        safe_mode=False,
        compile=False
    )
    return model

@st.cache_data
def load_classes():
    with open("class_names.json", "r") as f:
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
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        st.subheader("Prediction")
        st.write(f"Predicted class: **{predicted_class}**")
        st.write(f"Model confidence: **{confidence:.2f}%**")

        st.warning(
            "This is an educational prediction, not a medical diagnosis. "
            "Consult a qualified dermatologist for medical advice."
        )

