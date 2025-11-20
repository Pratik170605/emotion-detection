from flask import Flask, render_template, jsonify
import cv2
from keras.models import model_from_json
import numpy as np
from collections import deque
import threading

app = Flask(__name__)

# Load Emotion Model
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.keras")

# Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion Labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Emotion History
emotion_history = deque(maxlen=10)
confidence_history = deque(maxlen=10)

# Preprocessing Functions
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

# Start Webcam Thread
def start_webcam():
    webcam = cv2.VideoCapture(1)  # Change to 0 if needed
    while True:
        success, frame = webcam.read()
        if not success:
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)

        # Preprocess the frame for emotion detection
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

                    # Update emotion history and confidence history
                    emotion_history.append(label)
                    confidence_history.append(confidence)

                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Display the frame with the detected emotions
        cv2.imshow("Emotion Detector", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

# Start the webcam in a separate thread
threading.Thread(target=start_webcam, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotion_data')
def emotion_data():
    return jsonify({
        'emotions': list(emotion_history),
        'confidences': list(confidence_history)
    })

if __name__ == '__main__':
    app.run(debug=True)