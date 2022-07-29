From basic image processing towards object detection

The goal of this individual assignment was to apply some basic image processing techniques
to detect an object and produce a one minute video. 
The Video can be found [here]().

The video is structured as followes:  
* 0s-20s: Basic image processing techniques to provide special effects.  
      * Color and grayscale (±4s).  
      * Smoothing or blurring to reduce noise. Experiments with Gaussian and bi-lateral filters.   
      * Object grabbing in RGB and HSV color space.   

* 20s-40s: Round colored object as the key figure and perform some object detection.  
      * Sobel edge detector to detect horizontal and vertical edges.   
      * Hough transform.   
      * Gray scale projection, with the intensity values proportional to the likelihood of the object of interest being at that location (±2s+3s). Thus white means that (the center of) the object is at that particular location with 100 % certainty, while black means the opposite.

* 40s-60s: Ball detection and Face Detection.
      
For this project, Python and the OpenCV library were used for the implementation.   
The actual video captures were done using my smartphone.  

The skeleton python code for reading and writing video files was done using OpenCV in this repository: https://github.com/gourie/opencv_video.

For quicker processing, the input video was downsampled right from the start. 
