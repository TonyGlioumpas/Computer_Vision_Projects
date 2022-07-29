"""
Individual Assignment1_part4 - Computer Vision
Student: Antonios Glioumpas

Carte_Blanche_Ball_Detection.py : 
Tracks a green/yellow ball in real time from the camera or from video, doesn't record video. 
The script was used to perform ball detection on internal camera capture.

Based on "opencv_video" repository : https://github.com/gourie/opencv_video
"""

"""Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

def nothing():
    pass

# Use of Trackbar method to define the upper and lower RGB values of the ball-to-be-detected
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - B", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - G", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - R", "Trackbars", 0, 255, nothing)

cv2.createTrackbar("U - B", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - G", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - R", "Trackbars", 255, 255, nothing)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())


pts = deque(maxlen=args["buffer"])
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=1000)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	
	# define the lower and upper boundaries of the "green"
	# ball in the HSV color space, then initialize the
	# list of tracked points
	l_b = cv2.getTrackbarPos("L - B", "Trackbars")
	l_g = cv2.getTrackbarPos("L - G", "Trackbars")
	l_r = cv2.getTrackbarPos("L - R", "Trackbars")

	u_b = cv2.getTrackbarPos("U - B", "Trackbars") 
	u_g = cv2.getTrackbarPos("U - G", "Trackbars")      
	u_r = cv2.getTrackbarPos("U - R", "Trackbars")

	greenLower = (l_b, l_g, l_r)
	greenUpper = (u_b,  u_g, u_r)
	# Best Result:
	#greenLower = (23, 86, 7)
	#greenUpper = (103, 255, 255)
	
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
	# update the points queue
	pts.appendleft(center)

    	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		#cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	
	frame = cv2.putText(frame, 'Ball detection using cv2.findContours()', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
	
	# show the frame to our screen
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()