import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image

# -------------------------------
# Load CNN Model
# -------------------------------
model = tf.keras.models.load_model("model/emotion_model.keras")

# Emotion Labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# -------------------------------
# Load Songs
# -------------------------------
with open("songs.json", "r") as file:
    songs = json.load(file)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Emotion Based Music Recommendation", page_icon="🎵")

st.title("🎵 Emotion Based Music Recommendation System")
st.write("Upload a face image to detect emotion and get a song recommendation.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        st.error("No face detected.")
    else:

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (48, 48))

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=-1)

        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)

        emotion = emotion_labels[np.argmax(prediction)]

        st.success(f"Detected Emotion: {emotion}")

        emotion_key = emotion.lower()


if emotion_key in songs:
    st.success(f"🎵 Recommended Song: {np.random.choice(songs[emotion_key])}")
else:
    st.warning("No matching emotion found in songs.json")