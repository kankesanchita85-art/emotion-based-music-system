import json
import random

with open("songs.json", "r") as file:
    songs = json.load(file)

last_song = ""

def recommend_song(emotion):
    global last_song

    emotion = emotion.lower()

    if emotion in songs:

        available = songs[emotion]

        if len(available) == 1:
            last_song = available[0]
            return last_song

        while True:
            song = random.choice(available)

            if song != last_song:
                last_song = song
                return song

    return "No Song Found"