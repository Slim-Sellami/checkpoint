import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

# =========================
# Load model
# =========================
model = keras.models.load_model("model.keras", compile=False)

# =========================
# UI
# =========================
st.title("🐱🐶 Cats vs Dogs Classifier")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# =========================
# Preprocessing function
# =========================
def preprocess_image(image):
    image = image.resize((150, 150))  # must match training
    img_array = np.array(image)

    # ensure 3 channels
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array

# =========================
# Prediction
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)

    prediction = model.predict(processed)[0][0]

    # binary classification logic
    if prediction > 0.5:
        label = "🐶 Dog"
    else:
        label = "🐱 Cat"

    confidence = prediction if prediction > 0.5 else 1 - prediction

    # =========================
    # Output
    # =========================
    st.subheader("Result")
    st.write("Prediction:", label)
    st.write("Confidence:", f"{confidence:.2f}")