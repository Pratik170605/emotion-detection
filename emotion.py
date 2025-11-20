import cv2
from keras.models import model_from_json
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import pyttsx3

# --- Load Emotion Model ---
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.keras")

# --- Haar Cascade ---
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# --- Emotion Labels ---
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# --- Emotion History for Chart ---
emotion_history = deque(maxlen=10)
confidence_history = deque(maxlen=10)

# --- Voice Setup ---
engine = pyttsx3.init()
spoken_emotions = deque(maxlen=5)  # avoid repeating voice

def speak_emotion(emotion):
    if emotion not in spoken_emotions:
        engine.say(f"You seem {emotion} today!")
        engine.runAndWait()
        spoken_emotions.append(emotion)

# --- Preprocessing Functions ---
def extract_features(image):
    image = np.array(image)
    image = image.reshape(1, 48, 48, 1)
    return image / 255.0

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

# --- Start Webcam ---
webcam = cv2.VideoCapture(1)  # if not working try 0

while True:
    success, frame = webcam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    gray = preprocess_frame(frame)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            features = extract_features(face)
            prediction = model.predict(features)[0]
            label_index = np.argmax(prediction)
            label = labels[label_index]
            confidence = prediction[label_index] * 100

            if confidence > 50:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
                text = f"{label} ({confidence:.1f}%)"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                emotion_history.append(label)
                confidence_history.append(confidence)

                speak_emotion(label)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Emotion Detector", frame)

    # Press 'c' to show chart of last 10 emotions
    if cv2.waitKey(1) & 0xFF == ord('c'):
        if emotion_history:
            plt.clf()
            plt.bar(emotion_history, confidence_history, color='skyblue')
            plt.title("Last 10 Detected Emotions")
            plt.ylabel("Confidence (%)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
