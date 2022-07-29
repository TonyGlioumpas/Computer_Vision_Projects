"""
Individual Assignment1_part3 - Computer Vision
Student: Antonios Glioumpas

Likelihood_Map.py : 
Performs Template Matching on imput video frames with template image and shows:
- the original color frames with a box around the object of the template
- the likelihood map (intensity values proportional to the likelihood of the object of interest being at that location)

Example Execution
shell script:
   $ python Likelihood_Map.py -i <PATH_TO_INPUT_VIDEO_FILE> -o <PATH_TO_OUPUT_VIDEO_FILE>
   
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
import matplotlib as plt

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
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height),0)

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            
            # Read template image from path
            template = cv2.imread('Template.jpg',0)
            h2,w2 = template.shape[:2]
                    
            #resize video frame dimensions
            scale_percent =60 # percentage of original video frame size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame = cv2.resize(frame , dsize = dim)

            #resize template dimensions
            scale_percent = 16# percentage of original template image size
            width = int(template.shape[1] * scale_percent / 100)
            height = int(template.shape[0] * scale_percent / 100)
            dim = (width, height)
            template = cv2.resize(template , dsize = dim)
            
            method0 = eval('cv2.TM_SQDIFF')
            method1 = eval('cv2.TM_SQDIFF_NORMED')
            method2 = eval('cv2.TM_CCORR ')
            method3 = eval('cv2.TM_CCORR_NORMED')
            method4 = eval('cv2.TM_CCOEFF')
            method5 = eval('cv2.TM_CCOEFF_NORMED')

            method = method5

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # template match
            res = cv2.matchTemplate(gray,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Draw box around the Max likelihood location of the template
            top_left = max_loc
            h,w =template.shape[:2]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(gray,top_left, bottom_right, (255,0,0), 2)
            
            res = cv2.putText(res,'Likelihood Map from Template Matching', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA) 
            
            frame = cv2.putText(frame,'Original Shot - Box around object', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA) 
            cv2.rectangle(frame,top_left, bottom_right, (51,255,51), 2)
            cv2.imshow("Color Video Feed - Box",frame)

            #cv2.imshow("Grayscale Video Feed - Box",gray)
            
            cv2.imshow("Likelihood Map",res) 
            
            #plt.imshow(res,cmap = 'gray')  
            #plt.show()
            
            out.write(res)     
                        
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
