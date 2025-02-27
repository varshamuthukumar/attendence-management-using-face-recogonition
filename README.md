# attendence-management-using-face-recogonition
import cv2
import face_recognition
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Load known faces and their names
known_face_encodings = []
known_face_names = []

# Load the known faces from a folder
def load_known_faces():
    images_path = "known_faces/"
    for filename in os.listdir(images_path):
        image = face_recognition.load_image_file(f"{images_path}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(filename.split('.')[0])

load_known_faces()

# Initialize variables
face_locations = []
face_encodings = []
face_names = []

# Load the attendance record
attendance_file = "attendance.csv"

# Create a DataFrame if the attendance file doesn't exist
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Check if the name is already recorded for the day
    df = pd.read_csv(attendance_file)
    if not any(df['Name'] == name) or not any(df['Date'] == date_str):
        df = df.append({"Name": name, "Date": date_str, "Time": time_str}, ignore_index=True)
        df.to_csv(attendance_file, index=False)

# Run the video capture and face recognition
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Resizing for faster processing

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_names = []

    # Loop through each face found in the frame
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # If a match is found, use the name associated with that face encoding
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        face_names.append(name)

        # Mark attendance if a recognized face is found
        if name != "Unknown":
            mark_attendance(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

    # Show the video feed
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
