"""
Individual Assignment1_part1 - Computer Vision
Student: Antonios Glioumpas

Part1:
- Color / Grayscale Conversion
- Gaussian Filtering
- Bilateral Filtering
- Grab object in RBG / HSV
- Erosion / Dilation
- Color Mask

To run the Assignment1_part1.py script
shell script:
   $ python Assignment1_part1.py -i <PATH_TO_INPUT_VIDEO_FILE> -o <PATH_TO_OUPUT_VIDEO_FILE>

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

def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
   
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height)) # to create a VideoWriter object

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            # 2.1 COLOR/GREYSCALE ---------------------
            if between(cap, 0, 1000):
                frame = cv2.putText(frame, 'Color', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
            if between(cap, 1000, 2000):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.putText(frame, 'GreyScale', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Fix for the dropping frames issue
            if between(cap, 2000, 3000):
                frame = cv2.putText(frame, 'Color', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)    
            if between(cap, 3000, 4000):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.putText(frame, 'GreyScale', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Fix for the dropping frames issue

            # 2.2 SMOOTHING(BLURRING) -------------------
            if between(cap, 4000, 4500):
                frame = cv2.putText(frame, 'No Smoothing', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

            if between(cap, 4500, 5500):
               frame = cv2.GaussianBlur(frame, ksize=(7,7),sigmaX=1, sigmaY=1)
               frame = cv2.putText(frame, 'Gaussian Filter, kernel size=7', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
            if between(cap, 5500, 7500):
                frame = cv2.GaussianBlur(frame, ksize=(15,15),sigmaX=1, sigmaY=1)
                frame = cv2.putText(frame, 'Gaussian Filter, kernel size=15', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
            if between(cap, 7500, 9500):
                frame = cv2.bilateralFilter(frame,7,75,75)
                frame = cv2.putText(frame, 'Bilateral Filter, size: 7', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
            if between(cap, 9500, 12000):
                frame = cv2.bilateralFilter(frame,15,75,75)
                frame = cv2.putText(frame, 'Bilateral Filter, size: 15', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)    

            # 2.3 GRAB OBJECT RGB & HSV - MORPHOLOGICAL OPERATOR --------------------
            if between(cap, 12000, 13000):
                frame = cv2.putText(frame, 'Original Shot', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA) 
            
            if between(cap, 13000, 15000):
                lower_blue = np.array([82,0,0]) #BGR
                upper_blue = np.array([255,143,61])
                mask_rgb = cv2.inRange(frame,lower_blue,upper_blue)
                frame = cv2.inRange(frame,lower_blue,upper_blue)
                frame = cv2.putText(frame, 'Grab in RBG', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)  
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Fix for the dropping frames issue

            if between(cap, 15000, 17000):
                hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                lower_blue = np.array([100,100,0])
                upper_blue = np.array([180,230,255])
                frame = cv2.inRange(hsv,lower_blue,upper_blue)
                frame = cv2.putText(frame, 'Grab in HSV', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Fix for the dropping frames issue

            # To get a better grab of the object, Erosion followed by Dilation is used
            if between(cap, 17000, 18500):
                hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                lower_blue = np.array([100,100,0])
                upper_blue = np.array([180,230,255])
                frame = cv2.inRange(hsv,lower_blue,upper_blue)
                # To eliminate background specs with same color as the ROI
                kernel_erosion = np.ones((5,5),np.uint8)
                erosion = cv2.erode(frame,kernel_erosion,iterations = 1)
                frame = erosion
                # To close gaps
                kernel_closing = np.ones((15,15),np.uint8)
                closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel_closing)
                frame = closing
                frame = cv2.putText(frame, 'Dilation & Erosion kernel size=15', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA) 
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Fix for the dropping frames issue

            if between(cap, 18500, 20000):
                lower_blue = np.array([82,0,0]) #BGR
                upper_blue = np.array([255,143,61])
                mask_rgb = cv2.inRange(frame,lower_blue,upper_blue)
                frame = cv2.bitwise_and(frame, frame, mask=mask_rgb)
                frame = cv2.putText(frame, 'Color Mask using Bitwise AND', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA) 

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)

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
