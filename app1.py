import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time

# Load model
model = load_model("exibition_recognition_model.h5")

# Class names (assume 0 = Non-Food, 1 = Food, and more subclasses for food categories)
class_names = ['Non-Food', 'Bread', 'Dairy product', 'Dessert', 'Egg',
               'Fried food', 'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Vegetable']

# Image preprocessing
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Streamlit layout
st.set_page_config(page_title="Live Food vs Non-Food Classifier", layout="centered")
st.title("🎥 Real-time Food vs Non-Food Classifier")

# Option for webcam or image upload
mode = st.radio("Choose a mode", ["Webcam", "Upload Image"])

if mode == "Webcam":
    # Start Webcam Prediction
    run = st.checkbox('Start Webcam Prediction')

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("⚠️ Unable to access webcam.")
        else:
            FRAME_WINDOW = st.image([])

            st.info("Press 'Stop Webcam Prediction' to end the session.")
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("⚠️ Failed to access webcam.")
                    break
                
                # Predict
                input_data = preprocess_image(frame)
                prediction = model.predict(input_data)
                top_class_idx = np.argmax(prediction[0])
                confidence = prediction[0][top_class_idx] * 100
                label = class_names[top_class_idx]

                # Threshold for confidence, assume 50% confidence threshold for food vs non-food
                if confidence < 50:
                    label = "Non-Food"
                    confidence = 100 - confidence

                # Label color based on classification
                color = (0, 255, 0) if label != "Non-Food" else (0, 0, 255)

                # Annotate frame with label and confidence
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, f"{label} ({confidence:.2f}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Show the annotated frame
                FRAME_WINDOW.image(annotated_frame)
                time.sleep(0.5)  # Adjust sleep time for performance

            cap.release()
            st.success("Webcam session ended.")
    else:
        st.warning("Enable webcam by checking the box above.")
        
elif mode == "Upload Image":
    # Image upload section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image for model prediction
        img_array = np.array(image)
        input_data = preprocess_image(img_array)

        # Predict using the model
        prediction = model.predict(input_data)
        top_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][top_class_idx] * 100
        label = class_names[top_class_idx]

        # Threshold for confidence
        if confidence < 50:
            label = "Non-Food"
            confidence = 100 - confidence

        # Display prediction
        st.write(f"Prediction: {label} with confidence of {confidence:.2f}%")
        
        # Show the predicted label with confidence
        st.write(f"Prediction: {label} ({confidence:.2f}%)")