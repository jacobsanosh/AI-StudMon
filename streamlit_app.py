# import the required libraries
import numpy as np
import cv2
import face_recognition
import os
from keras.models import load_model
import streamlit as st
from PIL import Image, ImageDraw
import db_connect
from keras.preprocessing.image import img_to_array
from datetime import datetime, timedelta

# Define the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load model
classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.h5")
#connect to db
connection=db_connect.connect_to_supabase()
cur=connection[0]
conn=connection[1]
# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

# Define the directory containing student images
directory = "./students"
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
def facerec(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb) 
    face_locations = face_recognition.face_locations(frame_rgb)

    # If no faces are detected, return "Unknown"
    if len(face_locations) == 0:
        return "Unknown"
    
    # Extract the face encoding for the first (and only) face detected
    top, right, bottom, left = face_locations[0]
    face_encoding = face_recognition.face_encodings(frame_rgb, [(top, right, bottom, left)])[0]
    
    # Compare the face encoding with known face encodings
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    name="unknown"
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        
    return name

    
def create_table_if_not_exists(table_name):
    query = f"CREATE TABLE IF NOT EXISTS {table_name} (student_name varchar(20), emotion varchar(20), timestamp timestamp)"
    try:
        cur.execute(query)
        conn.commit()
        print("Table created successfully.")

    except Exception as e:
        print("Error:", e)
    
def insert_data_into_table(table_name, student_name, emotion, timestamp):
    query = f"INSERT INTO {table_name} (student_name, emotion, timestamp) VALUES ('{student_name}', '{emotion}', '{timestamp}')"
    try:
        cur.execute(query)
        conn.commit()
        print("Table created successfully.")

    except Exception as e:
        print("Error:", e)

def main():
    # Face Analysis Application
    st.markdown("<h1 style='text-align: center; color: white; line-height: 0.5;'>🤖</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: orange;  line-height: 0;'>Studmon.ai</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: white; font-weight: lighter;'>AI Based Student Monitoring System</h4>", unsafe_allow_html=True)

    activities = ["Home", "Live Face Emotion Detection", "Dashboard", "About Us"]
    choice = st.sidebar.radio("Select Activity:", activities)

    # Homepage
    if choice == "Home":

        # html_temp_home1 = """</br></br>
        #                      <div style="background-color:#FC4C02;padding:0.5px">
        #                      <h4 style="color:white;text-align:center;">
        #                      Start Your Real Time Face Emotion Detection.
        #                      </h4>
        #                      </div>
        #                      </br>"""
        
        html_temp_home1 = """<hr></br> 
                             <div style="display:flex; flex-direction: column; align-items: center;">
                             <p style='text-align: center; color: white; font-family: Courier; font-weight: lighter;'>Experience the future of education with our AI-enhanced student monitoring and understanding assessment system. Personalize learning, improve academic outcomes, and optimize resource allocation...✨</p>
                             </br>
                             <p style='text-align: center; color: white; font-family: Courier; font-weight: lighter;'>📊 Assess how well students understand their lessons💡& </br>how they're feeling 😊.</p>
                             </br>
                             <div style="background-color:#FC9C01;padding:0.5px;">
                             <h4 style="color:white;text-align:center;">
                             Start Emotion Detection. 
                             </h4>
                             </div>
                             <div>
                             </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        # st.write("""
        # 1. Click the dropdown list in the top left corner and select Live Face Emotion Detection.
        # 2. This takes you to a page which will tell if it recognizes your emotions.
        # """)

    # Live Face Emotion Detection
    elif choice == "Live Face Emotion Detection":
        st.header("Webcam Live Feed")
        st.subheader('''
        Welcome to the other side of the SCREEN!!!
        * Get ready with all the emotions you can express. 
        ''')
        if st.button("Start Class"):
        # Initialize OpenCV video capture
          table_name = "class_" + datetime.now().strftime("%Y%m%d_%H%M%S")
          create_table_if_not_exists(table_name)
          cap = cv2.VideoCapture(0)
          start_time = datetime.now()
          faces_data=[]
          while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            else:
                if (datetime.now() - start_time).seconds >= 2:
                    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                    image=img_gray, scaleFactor=1.3, minNeighbors=5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img=frame, pt1=(x, y), pt2=(
                            x + w, y + h), color=(0, 255, 255), thickness=2)
                        roi_gray = img_gray[y:y + h , x:x + w]
                        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)  
                        if np.sum([roi_gray]) != 0:
                            roi = roi_gray.astype('float') / 255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi, axis=0)
                            prediction = classifier.predict(roi)[0]
                            maxindex = int(np.argmax(prediction))
                            finalout = emotion_labels[maxindex]
                            output = str(finalout)
                            label_position = (x, y-10)
                            name=facerec(frame[y:y+h +4, x:x+w])  
                            
                            cv2.putText(frame, f'{output}{name} ', label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
                            faces_data.append((name, output, datetime.now()))

                    
                            for data in faces_data:
                                name, output, timestamp = data
                                insert_data_into_table(table_name, name, output, timestamp)
                            faces_data=[]    
                            start_time=datetime.now()
                #for processing
                if (datetime.now() - procssingTime).seconds >= 3:
                    groupEmotion.processingEmotion(table_name)
                    start_time=datetime.now()
                
            cv2.imshow('Face Detection, Emotion Detection, and Recognition', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

          cap.release()
          cv2.destroyAllWindows()
    if connection:
         cur.close()
         conn.close()    

    # Dashboard
    elif choice == "Dashboard":
        st.subheader("Dashboard")
        html_temp_about1 = """<div style="background-color:#36454F;padding:30px">
                                    <h4 style="color:white;">
                                     This app predicts facial emotion using a Convolutional neural network.
                                     Which is built using Keras and Tensorflow libraries.
                                     Face detection is achieved through face_recognition library.
                                    </h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)
    # About
    elif choice == "About Us":
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
    
    

if __name__ == "__main__":
    main()