import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("emotion_model.h5")

# Define emotion labels (must match model output)
class_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("😊 Face Emotion Detection System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB"))
    st.image(img_array, caption="Uploaded Image", use_column_width=True)

    # Load Haarcascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("😕 No face detected in the image.")
    else:
        for (x, y, w, h) in faces:
            # Extract and preprocess face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

            # Predict emotion
            prediction = model.predict(roi_reshaped)
            emotion_index = np.argmax(prediction)

            # ✅ Safe indexing to avoid list index errors
            if emotion_index < len(class_labels):
                emotion_label = class_labels[emotion_index]
            else:
                emotion_label = "Unknown"

            # Draw rectangle + label
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_array, emotion_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        st.image(img_array, caption="Detected Emotion", use_column_width=True)
        st.success(f"Predicted Emotion: **{emotion_label}**")