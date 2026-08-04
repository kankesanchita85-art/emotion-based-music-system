import random

def chatbot_response(user_input):

    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return "👋 Hello! Welcome to the Emotion Based Music Recommendation System."

    elif "project" in user_input:
        return ("This project detects human emotions using Deep Learning (CNN) "
                "and recommends songs based on the detected emotion.")

    elif "cnn" in user_input:
        return ("CNN (Convolutional Neural Network) is a Deep Learning model "
                "used for image classification and emotion detection.")

    elif "dataset" in user_input:
        return ("This project uses the FER-2013 dataset containing "
                "35,887 facial expression images of seven emotions.")

    elif "accuracy" in user_input:
        return ("The current trained model achieves approximately "
                "63.5% validation accuracy on the FER-2013 dataset.")

    elif "emotion" in user_input:
        return ("The system detects:\n"
                "😊 Happy\n"
                "😢 Sad\n"
                "😠 Angry\n"
                "😨 Fear\n"
                "😐 Neutral\n"
                "😲 Surprise\n"
                "🤢 Disgust")

    elif "technology" in user_input or "technologies" in user_input:
        return ("Technologies Used:\n"
                "• Python\n"
                "• TensorFlow\n"
                "• Keras\n"
                "• OpenCV\n"
                "• Streamlit\n"
                "• NumPy")

    elif "music" in user_input:
        return ("After detecting the user's emotion, "
                "the system recommends songs from a predefined music library.")

    elif "developer" in user_input or "developed" in user_input:
        return ("This project was developed as a Deep Learning project "
                "for Emotion Based Music Recommendation.")

    elif "thank" in user_input:
        return "😊 You're welcome! Have a great day."

    else:
        replies = [
            "I'm sorry, I don't understand your question.",
            "Can you ask your question differently?",
            "Please ask me about the project, CNN, emotions, accuracy or dataset.",
            "I can answer questions related to this project."
        ]

        return random.choice(replies)