"""
Individual Assignment1_part3 - Computer Vision
Student: Antonios Glioumpas

Hough_transform.py : detects circular shapes in the frame and visualizes the detected circles by contours overlayed on the original color frame

Example execution
shell script:
   $ python Hough_transform.py -i <PATH_TO_INPUT_VIDEO_FILE> -o <PATH_TO_OUPUT_VIDEO_FILE>
   
Based on "opencv_video" repository : https://github.com/gourie/opencv_video
"""

"""Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys
import numpy as np

# helper function to change what you do based on video seconds
# arguments "lower: int" and "upper: int" are measured in milliseconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

def Circles(gray,rows,frame,par1,par2,minRad,maxRad):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=par1, param2=par2, minRadius=minRad, maxRadius=maxRad)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3)
            frame = cv2.putText(frame, 'Hough Circles '+'param1='+str(par1)+' param2='+str(par2)+' maxRadius='+str(maxRad)+' minRadius='+str(minRad), (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA) 
    return frame

def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
   
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            
            if between(cap, 0, 1000):
                frame = cv2.putText(frame, 'Original Shot', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            
            # 3.2 --- Hough Transform
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect circles in the image
            gray = cv2.medianBlur(gray, 5)
            rows = gray.shape[0]

            if between(cap, 1000, 4000):
                frame = Circles(gray,rows,frame,par1=41,par2=41,minRad=63,maxRad=100)
            if between(cap, 4000, 7000):
                frame = Circles(gray,rows,frame,par1=11,par2=23,minRad=50,maxRad=55)
            #if between(cap, 5000, 7000):
                #frame = Circles(gray,rows,frame,par1=41,par2=31,minRad=53,maxRad=100)
            if between(cap, 7000, 10000):
                frame = Circles(gray,rows,frame,par1=41,par2=41,minRad=53,maxRad=100)
            #if between(cap, 8000, 10000):
            #   frame = Circles(gray,rows,frame,par1=44,par2=44,minRad=54,maxRad=100)

            # write frame that you processed to output    
            out.write(frame)           
    
            imS = cv2.resize(frame, (1600, 900))       # Resize image
            cv2.imshow("output", imS)
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
