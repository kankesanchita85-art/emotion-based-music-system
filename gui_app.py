import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
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

# Main Window
root = tk.Tk()
root.title("Emotion Based Music Recommendation System")
root.geometry("1000x700")
root.configure(bg="#EAF4FC")

title = tk.Label(
    root,
    text="🎵 Emotion Based Music Recommendation System",
    font=("Arial", 22, "bold"),
    bg="#EAF4FC",
    fg="#0B3D91"
)

title.pack(pady=15)

video_label = tk.Label(root)
video_label.pack()

emotion_var = tk.StringVar(value="Emotion : --")
confidence_var = tk.StringVar(value="Confidence : --")
song_var = tk.StringVar(value="Song : --")

emotion_label = tk.Label(
    root,
    textvariable=emotion_var,
    font=("Arial",16,"bold"),
    bg="#EAF4FC"
)
emotion_label.pack()

confidence_label = tk.Label(
    root,
    textvariable=confidence_var,
    font=("Arial",16),
    bg="#EAF4FC"
)
confidence_label.pack()

song_label = tk.Label(
    root,
    textvariable=song_var,
    font=("Arial",16),
    bg="#EAF4FC"
)
song_label.pack()

cap = cv2.VideoCapture(0)
def update_frame():

    ret, frame = cap.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)

            index = np.argmax(prediction)

            emotion = emotion_labels[index]

            confidence = prediction[0][index] * 100

            song = recommend_song(emotion)

            emotion_var.set(f"😊 Emotion : {emotion}")
            confidence_var.set(f"📊 Confidence : {confidence:.2f}%")
            song_var.set(f"🎵 Song : {song}")

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)

        img = img.resize((700,500))

        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk

        video_label.configure(image=imgtk)

        root.after(10, update_frame)


def start_camera():
    update_frame()


def stop_camera():
    global cap

    if cap.isOpened():
        cap.release()

    video_label.configure(image="")


def exit_app():

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()

    root.destroy()


button_frame = tk.Frame(root, bg="#EAF4FC")
button_frame.pack(pady=20)

start_button = ttk.Button(
    button_frame,
    text="▶ Start Camera",
    command=start_camera
)
start_button.grid(row=0, column=0, padx=10)

stop_button = ttk.Button(
    button_frame,
    text="⏹ Stop Camera",
    command=stop_camera
)
stop_button.grid(row=0, column=1, padx=10)

exit_button = ttk.Button(
    button_frame,
    text="❌ Exit",
    command=exit_app
)
exit_button.grid(row=0, column=2, padx=10)

# Start camera automatically
update_frame()

root.mainloop()