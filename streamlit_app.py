# import the required libraries
import numpy as np
import cv2
import face_recognition
import os
from keras.models import load_model
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import db_connect
from keras.preprocessing.image import img_to_array
from datetime import datetime, timedelta
import groupEmotion,dashboard,analytics

from supabase_py import create_client
from graph import derive_graph

# Create a Supabase client
supabase_url = "https://cmjrimwbnzszaozkkcam.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNtanJpbXdibnpzemFvemtrY2FtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTM4MDQzMTMsImV4cCI6MjAyOTM4MDMxM30.rfkhuoLt3d_zr2xUaDXZ3k9uCPutT-mIAaJUD3E565k"
supabase = create_client(supabase_url, supabase_key)



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
    st.set_page_config(layout="wide")
    tables = dashboard.dashboard(cur)
    emoji_mapping = {
        'Happy': '😊',
        'Neutral': '😐',
        'Fear': '😨',
        'Sad': '😞',
        'Surprise': '😮'
    }
    # Face Analysis Application
    st.markdown("<h1 style='text-align: center; color: white; line-height: 0.5;'>🤖</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: orange;  line-height: 0;'>Studmon.ai</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: white; font-weight: lighter;'>AI Based Student Monitoring System</h4>", unsafe_allow_html=True)

    activities = ["Home", "Live Face Emotion Detection", "Dashboard", "About Us"]
    choice = st.sidebar.radio("Select Activity:", activities)

    # Homepage
    if choice == "Home":
        html_temp_home1 = """<hr></br> 
                             <div style="display:flex; flex-direction: column; align-items: center;">
                             <p style='text-align: center; color: white; font-family: Courier; font-weight: lighter;'>Experience the future of education with our AI-enhanced student monitoring and understanding assessment system. Personalize learning, improve academic outcomes, and optimize resource allocation...✨</p>
                             </br>
                             <p style='text-align: center; color: white; font-family: Courier; font-weight: lighter;'>📊 Assess how well students understand their lessons💡& </br>how they're feeling 😊.</p>
                             </br>
                             <div style="background-color:#FC9C01;padding:0.5px;">
                             <h4 style="color:white;text-align:center;">
                             Start Emotion Detection
                             </h4>
                             </div>
                             <div>
                             </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

    # Live Face Emotion Detection
    elif choice == "Live Face Emotion Detection":
        # st.header("Webcam Live Feed")
        # st.subheader('''
        # Welcome to the other side of the SCREEN!!!
        # * Get ready with all the emotions you can express. 
        # ''')


        css = """
            <style>
            .stButton>button {
                height: 4rem;
                margin: 0 auto;
                font-size: 18px;
               }
            </style>
             """
        st.markdown(css, unsafe_allow_html=True)

        html_temp_emotion_detection = """
                                      <hr></br> 
                                      <div style="display:flex; flex-direction: column; align-items: center;">
                                        <p style='text-align: center; color: white; font-family: Courier; font-weight: lighter;'>
                                             Monitor 🤓students' emotions in real-time🔄 during your class sessions.
                                            <br/>
                                            Start recording🎥 now to analyze their engagement levels🧠 and understanding💁...
                                        </p>
                                      </br>
                                      </div>
                                      </br>
                                      """
        st.markdown(html_temp_emotion_detection, unsafe_allow_html=True)
        
        if st.button("🔴 Start Class", use_container_width=True):
        # Initialize OpenCV video capture
          table_name = "class_" + datetime.now().strftime("%Y%m%d_%H%M%S")
          create_table_if_not_exists(table_name)
          cap = cv2.VideoCapture(0)
          start_time = datetime.now()
          procssingTime=start_time
          faces_data=[]
          while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            else:
                if (datetime.now() - start_time).seconds >= 15:
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
                if (datetime.now() - procssingTime).seconds >= 45:
                    groupEmotion.processingEmotion(cur,table_name,datetime.now())
                    procssingTime=datetime.now()
                
            cv2.imshow('Face Detection, Emotion Detection, and Recognition', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

          cap.release()
          cv2.destroyAllWindows()
  

 # Dashboard
    elif choice == "Dashboard":
        try:
            st.header("Dashboard")
            table_names = [table[0] for table in tables]
            selected_table = st.selectbox("Select Class ", table_names)
            if st.button("Derive Analytics",use_container_width=True):
                plot_bytes = derive_graph( selected_table,cur)
                st.image(plot_bytes)
                data_map=analytics.derive_analytics(selected_table,cur)
                if not bool(data_map)==True:
                    st.write("No data recorded !")
                else:
                    print(selected_table,":",data_map)
                    # Display class data
                    st.header("Class Data")
                    html_temp_d1 = """<hr>"""
                    st.markdown(html_temp_d1, unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("Emotions Percentages:")
                        labels = list(data_map['class_data']['emotions_percentages'].keys())
                        sizes = list(data_map['class_data']['emotions_percentages'].values())
                        fig1, ax1 = plt.subplots()
                        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                        ax1.axis('equal')
                        st.pyplot(fig1)
                    with col2:
                        st.write("Class Data:")
                        prominent_emotion = data_map['class_data']['prominent_class_emotion']
                        prominent_emoji = emoji_mapping.get(prominent_emotion, '')
                        st.write(f"Total Students: {data_map['class_data']['total_students']}")
                        st.write(f"Prominent Emotion: {prominent_emotion} {prominent_emoji}")
                        st.write(f"Avg Comprehension Score: {round(data_map['class_data']['avg_comprehension_score'],2)}%")
                        st.write(f"Avg Incomprehension Score: {round(data_map['class_data']['avg_incomprehension_score'],2)}%")

                    # st.write(f"Total Students: {data_map['class_data']['total_students']}")
                    # st.write("Emotions Percentages:")
                    # for emotion, percentage in data_map['class_data']['emotions_percentages'].items():
                    #     st.write(f"{emotion}: {percentage}%")
                    # st.write(f"Prominent Class Emotion: {data_map['class_data']['prominent_class_emotion']}")
                    # st.write(f"Comprehension Score: {data_map['class_data']['comprehension_score']}%")
                    # st.write(f"Average Comprehension Score: {data_map['class_data']['avg_comprehension_score']}%")
                    # st.write(f"Average Incomprehension Score: {data_map['class_data']['avg_incomprehension_score']}%")

                    # Display student data
                    st.markdown(html_temp_d1, unsafe_allow_html=True)
                    st.header("Student Reports")
                    st.markdown(html_temp_d1, unsafe_allow_html=True)


                    for student_report in data_map['student_reports']:
                        cl1, cl2 = st.columns([1, 1])
                        
                        # Emotions pie chart
                        with cl1:
                            st.subheader("Emotions Percentage:")
                            labels = student_report['emotions_percentage'].keys()
                            sizes = student_report['emotions_percentage'].values()
                            fig1, ax1 = plt.subplots()
                            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                            ax1.axis('equal')
                            st.pyplot(fig1)
                        
                        # Other data
                        with cl2:
                            st.subheader(f"Student: {student_report['student']}")
                            prominent_emotion = data_map['class_data']['prominent_class_emotion']
                            prominent_emoji = emoji_mapping.get(prominent_emotion, '')
                            st.write(f"Prominent Emotion: {prominent_emotion} {prominent_emoji}")
                            st.write(f"Understanding: {round(student_report['understanding'],2)}%")
                            st.write(f"Not Understanding: {round(student_report['not_understanding'],2)}%")
                        
                        st.markdown(html_temp_d1, unsafe_allow_html=True)



        except Exception as e:
            st.error(f"Error occurred: {e}")

    # About
    elif choice == "About Us":
        st.header("Studmon.ai")
        html_temp_about1 = """
        <div style="background-color:#36454F;padding:30px">
            <h6 style="color:white;font-family: Courier; font-weight: lighter;">            
                🤖Studmon.ai is an AI-based Student Monitoring System designed to assess students' comprehension and emotional states during lessons in real-time. </br></br>It utilizes cutting-edge technology, including facial expression analysis and emotion detection, to provide insights into students' understanding and engagement levels.</br> </br>🧠 Studmon.ai aims to revolutionize education by personalizing learning, improving academic outcomes, and optimizing resource allocation in educational institutions. 🚀
            </h6>
        </div>
        <br/>
        <hr/>
        <br/>
        

        """
        st.markdown(html_temp_about1, unsafe_allow_html=True)

    
        features_list = [
        "🧠 Real-time assessment of students' comprehension and emotional states during lessons",
        "🔔 Automatic notifications for concept reviews when needed",
        "🎓 Personalized learning experience",
        "📈 Improved academic outcomes",
        "💡 Optimized resource allocation",
        "🖥️ Robust and user-friendly interface",
        "📊 Comprehensive reports and analytics for educators and administrators",
        ]
    
        st.markdown("<h3 style='color: orange;'>Key Features:</h3>", unsafe_allow_html=True)
        st.markdown("<ul>", unsafe_allow_html=True)
        for feature in features_list:
            st.markdown(f"<li style='color: white;'>{feature}</li>", unsafe_allow_html=True)
        st.markdown("</ul><br/><hr/>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    if connection:
         cur.close()
         conn.close()  