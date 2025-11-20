import cv2
from keras.models import model_from_json
import numpy as np
import time

# Load the trained emotion detection model
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.keras")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels for emotion categories
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Function to preprocess face for emotion detection model
def extract_features(image):
    image = np.array(image)
    image = image.reshape(1, 48, 48, 1)  # Reshaping for the model's input
    return image / 255.0  # Normalize image to [0, 1] range

# Preprocessing the frame
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization (Improves brightness and contrast)
    gray = cv2.equalizeHist(gray)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Gaussian Blur (Reduce noise)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

# Set up webcam 
webcam = cv2.VideoCapture(0)


while True:
    success, frame = webcam.read()
    if not success:
        break

    # Mirror flip (optional)
    frame = cv2.flip(frame, 1)

    # Preprocess the frame (grayscale conversion, histogram equalization, etc.)
    gray = preprocess_frame(frame)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no face is detected, continue the loop
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            # Crop the face region and resize to 48x48 (model input size)
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))

            # Extract features for the model
            features = extract_features(face)

            # Predict the emotion using the model
            prediction = model.predict(features)[0]
            label_index = np.argmax(prediction)
            label = labels[label_index]
            confidence = prediction[label_index] * 100

            # Only show emotions with confidence 
            if confidence > 50:
                # Draw rectangle around face and put label with confidence
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
                text = f"{label} ({confidence:.1f}%)"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                # If the confidence is low, label as 'unknown'
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
       
    # Show the output frame
    cv2.imshow("Emotion Detector", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
