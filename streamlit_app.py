import numpy as np
import cv2
import face_recognition
import os
from keras.models import load_model
import streamlit as st
from PIL import Image

# Define the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load model
classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.h5")

# Define the directory containing student images
directory = "./Students"
known_face_encodings = []
known_face_names = []

# Load known face encodings and names
for filename in os.listdir(directory):
    if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
        image = face_recognition.load_image_file(os.path.join(directory, filename))
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Define a function to preprocess face images
def preprocess_face_image(face_image, target_size):
    face_image = face_image.convert("RGB").resize(target_size)
    image_array = np.array(face_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

def main():
    # Face Analysis Application
    st.title("AI based student monitoring system")
    activities = ["Home", "Live Face Emotion Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    # Homepage
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#FC4C02;padding:0.5px">
                             <h4 style="color:white;text-align:center;">
                             Start Your Real Time Face Emotion Detection.
                             </h4>
                             </div>
                             </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
        1. Click the dropdown list in the top left corner and select Live Face Emotion Detection.
        2. This takes you to a page which will tell if it recognizes your emotions.
        """)

    # Live Face Emotion Detection
    elif choice == "Live Face Emotion Detection":
        st.header("Webcam Live Feed")
        st.subheader('''
        Welcome to the other side of the SCREEN!!!
        * Get ready with all the emotions you can express. 
        ''')

        # Initialize OpenCV video capture
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(frame_rgb)

            for (top, right, bottom, left) in face_locations:
                face_image = frame_rgb[top:bottom, left:right]
                preprocessed_face = preprocess_face_image(Image.fromarray(face_image), (224, 224))
                data_face = np.expand_dims(preprocessed_face, axis=0)

                roi_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                emotion_prediction = classifier.predict(roi)[0]
                emotion_index = np.argmax(emotion_prediction)
                predicted_emotion = emotion_labels[emotion_index]

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    matched_index = matches.index(True)
                    name = known_face_names[matched_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f'{name} {predicted_emotion}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # st.image(frame, channels="BGR", use_column_width=True)

        cap.release()

    # About
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1 = """<div style="background-color:#36454F;padding:30px">
                                    <h4 style="color:white;">
                                     This app predicts facial emotion using a Convolutional neural network.
                                     Which is built using Keras and Tensorflow libraries.
                                     Face detection is achieved through face_recognition library.
                                    </h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

if _name_ == "_main_":
    main()