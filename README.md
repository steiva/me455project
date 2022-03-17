## Creating a disparity map using two android phones and a rubber band. 

This is a final project for the University of Washington CSE ME455 course, "Computer Vision". Participants:
Shaheryar Hasnain, Ivan Stepanov, Digjay Das.

The topic of this project is Stereo Depth perception using a mobile camera(s). Two android phones with similar
main lenses are used. We just tied them up with a rubber band and tried it out. You can do the same! 

Project code structure:
* dependencies.py - file storing core functions for rectification of images and their post-processing.
* get_frames.py - code lets you take multiple images for data collection. Hook up your IP cameras and create your dataset.
* main.py - input stereo images and get disparity from them. Uses two different methods, precalculated fundamental matrix for rectification (if available) and real time fundamental matrix calculation using feature matching. Gives a comparison of the two methods.
* real_time_depth.py - file runs disparity map calculation on real-time video data. It requires a precalculated stereoMap.xml file, which you can get by chessboard calibration, using the rectification.py.
* rectification.py - using multiple chessboard images from the same camera setup, it created a fundamental matrix and flushes it into stereoMap.xml. **IMPORTANT: LAUNCH THIS FILE FIRST, AS MOST OF THE PROJECT WILL NOT WORK WITHOUT IT.**
* sandbox.ipynb - alternative to main.py, which provides with quick testing and tweaking. Good for figuring out which code block does what and playing around.
* tweak_me.py - file takes in two RECTIFIED images, and lets you play around with the multitude of parameters that the stereoSGBM algorithm has, using sliders. A good way to get a feeling of what does what, or to determine the ideal settings for your problem.
* data/ - folders with images we used to find the disparity map on, as well as do the chessboard rectification.

Useful links that were used to complete this project.
* https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
* https://github.com/aryanvij02/StereoVision/tree/master/main_scripts
* https://www.andreasjakl.com/how-to-apply-stereo-matching-to-generate-depth-maps-part-3/
* https://github.com/niconielsen32/ComputerVision/blob/master/stereoVisionCalibration/stereovision_calibration.py
* https://www.youtube.com/watch?v=yKypaVl6qQo&ab_channel=TheCodingLib
* https://www.researchgate.net/figure/Pin-hole-camera-model-terminology-The-optical-center-pinhole-is-placed-at-the-origin_fig10_317498100
* https://docs.opencv.org/3.4/dd/d53/tutorial_py_depthmap.html
* https://learnopencv.com/depth-perception-using-stereo-camera-python-c/

![Disparity real-time gif](1_video.gif)