"""
Individual Assignment1_part2 - Computer Vision
Student: Antonios Glioumpas

Sobel_Edge_Detection.py : 
Performs edge detection on video frames using Sobel filtering, shows resulting frames and outputs filtered frames in output video.

Example Execution
shell script:
   $ python Sobel_Edge_Detection.py -i <PATH_TO_INPUT_VIDEO_FILE> -o <PATH_TO_OUPUT_VIDEO_FILE>
   
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
import skimage.exposure as exposure

# helper function to change what you do based on video seconds
# arguments "lower: int" and "upper: int" are measured in milliseconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

def Sobel(img_blur,ks,sc,frame):
    # apply sobel derivatives
    sobelx = cv2.Sobel(img_blur,cv2.CV_64F,dx=1,dy=0,ksize=ks,scale=sc)
    sobely = cv2.Sobel(img_blur,cv2.CV_64F,dx=0,dy=1,ksize=31,scale=5)
    # add and take square root
    sobel_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    # normalize to range 0 to 255 and clip negatives
    sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
    frame = sobel_magnitude
    frame = cv2.putText(frame, 'Sobel Edge Detection '+ 'ksize='+str(ks)+'scale='+str(5), (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Fix for the dropping frames issue
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
            # 3.1  ------# Sobel Edge Detection--------
            # Convert to graycsale
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Blur the image for better edge detection
            img_blur = cv2.GaussianBlur(img_gray, (0,0), 1.3, 1.3) 
                        
            #Filtered frame initialization            
            sob_frame = frame
  
            if between(cap, 0, 500):
                frame = cv2.putText(frame, 'Original Shot', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            if between(cap, 500, 2000):
                sob_frame = Sobel(img_blur,ks=31,sc=5,frame=frame)
            if between(cap, 2000, 3500):
                sob_frame = Sobel(img_blur,ks=15,sc=5,frame=frame)
            if between(cap, 3500, 5000):
                sob_frame = Sobel(img_blur,ks=3,sc=3,frame=frame)  

            im =  sob_frame        
            imS = cv2.resize(im, (1600, 900))       # Resize image
            cv2.imshow("output", imS)               
            
            # write frame that you processed to output
            out.write(sob_frame)

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
