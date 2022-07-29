"""
Individual Assignment1_part4 - Computer Vision
Student: Antonios Glioumpas

Face_Detection_camera.py : 
Tracks a face in real time from the camera. 

Based on "opencv_video" repository : https://github.com/gourie/opencv_video
"""
"""Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
"""
Example execution:
python Face_Detection_camera.py "D:\KU_Leuven\Computer_Vision\Assignment_1\haarcascade_frontalface_default.xml"
"""

import cv2
import sys
import time

def nothing(x):
    pass

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
fps = int(round(video_capture.get(5)))
#fps = int(fps/5) # to slow down recorded video
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
out = cv2.VideoWriter('FaceDetected.mp4', fourcc, fps, (frame_width, frame_height))

# Use of Trackbar method to tweek the parameters of the faceCascade.detectMultiScale
cv2.namedWindow("Trackbars")
cv2.createTrackbar("scaleFactor", "Trackbars", 1, 3, nothing)
cv2.createTrackbar("minNeighbors", "Trackbars", 5, 255, nothing)
cv2.createTrackbar("minSize", "Trackbars", 30, 255, nothing)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    scaleFac = cv2.getTrackbarPos("scaleFactor", "Trackbars")
    #print("scaleFactor",scaleFac)
    minNeigh = cv2.getTrackbarPos("minNeighbors", "Trackbars")
    #print("minNeighbors",minNeigh)
    minSiz = cv2.getTrackbarPos("minSize", "Trackbars")
    #print("minSize",minSiz)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors= int(minNeigh),
        minSize=(int(minSiz), int(minSiz)),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    frame = cv2.putText(frame, 'Face Detection using CascadeClassifier', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    out.write(frame)

    # Display the resulting frame
    #frame = cv2.resize(frame,(1600,900))  # Optional Resize of the output frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
out.release()
cv2.destroyAllWindows()
