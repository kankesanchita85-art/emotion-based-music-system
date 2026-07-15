import cv2
import numpy as np
import tensorflow as tf
from music_recommender import recommend_song

# Load CNN model
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

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (48,48))

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=-1)

        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)

        index = np.argmax(prediction)

        confidence = prediction[0][index] * 100

        emotion = emotion_labels[index]

        song = recommend_song(emotion)

        # Face Rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Emotion
        cv2.putText(
            frame,
            f"Emotion : {emotion}",
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        # Confidence
        cv2.putText(
            frame,
            f"Confidence : {confidence:.2f}%",
            (20,65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
            2
        )

        # Song Recommendation
        cv2.putText(
            frame,
            "Recommended Song:",
            (20,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            song,
            (20,130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

    cv2.imshow("Emotion Based Music Recommendation System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()