from chatbot import chatbot_response
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Emotion Based Music Recommendation",
    page_icon="🎵",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
model = tf.keras.models.load_model("model/emotion_model.keras")

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
# Title
# -------------------------------
st.title("🎵 Emotion Based Music Recommendation System")
st.write("Detect your emotion and get a music recommendation.")

# ===============================
# Upload OR Camera
# ===============================

st.subheader("📷 Choose Image Source")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("Or Take a Picture")

# If camera image exists, use it
if camera_image is not None:
    uploaded_file = camera_image

# ===============================
# Emotion Detection
# ===============================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Selected Image",
        use_container_width=True
    )

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:

        st.error("❌ No face detected.")

    else:

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (48, 48))

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=-1)

        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)

        confidence = np.max(prediction) * 100

        emotion = emotion_labels[np.argmax(prediction)]

        st.success(f"😊 Detected Emotion: {emotion}")

        st.info(f"Confidence: {confidence:.2f}%")

        emotion_key = emotion.lower()

        if emotion_key in songs:

            song = np.random.choice(songs[emotion_key])

            st.success("🎵 Recommended Song")

            st.markdown(f"## 🎧 {song}")

        else:

            st.warning("No song available for this emotion.")

# ===============================
# AI Chatbot
# ===============================

st.markdown("---")

st.header("🤖 AI Project Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

user_input = st.chat_input(
    "Ask me anything about this project..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    reply = chatbot_response(user_input)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )

    with st.chat_message("assistant"):
        st.write(reply)