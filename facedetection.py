import cv2
import numpy as np
import time

# Load the face detection classifier
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Camera could not be accessed")
    exit()

# Variable to check if 'q' key is pressed
q_key_pressed = False
while not q_key_pressed:
    ret, frame = video_capture.read()

    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the number of detected faces
        text = "Number of Faces Detected = " + str(len(faces))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (0, 30), font, 1, (255, 0, 0), 1)

        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            q_key_pressed = True
            break

# Release the camera and close windows
video_capture.release()
cv2.destroyAllWindows()
