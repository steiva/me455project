import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
import dependencies as deps
import time
import imutils

# When using an IP Webcam application, these would be the IP adresses of the
# cell phones you are using as your webcams. The left and right phones respectively.
base_url_L = 'http://10.19.203.50:8080'
base_url_R = 'http://10.18.199.231:8080'

# Create videoCapture objects for both video streams.
CamL = cv2.VideoCapture(base_url_L + '/video')
CamR = cv2.VideoCapture(base_url_R + '/video')

# Upload the precalculated fundamental matrix, obtained from rectification.py.
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Set parameters for the StereoSGBM algorithm.
minDisparity = 0
numDisparities = 128
blockSize = 5
disp12MaxDiff = 1
uniquenessRatio = 3
speckleWindowSize = 10
speckleRange = 12

# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                               numDisparities=numDisparities,
                               blockSize=blockSize,
                               disp12MaxDiff=disp12MaxDiff,
                               uniquenessRatio=uniquenessRatio,
                               speckleWindowSize=speckleWindowSize,
                               speckleRange=speckleRange
                               )
# Calculating disparith using the StereoSGBM algorithm
left_matcher = stereo

# Creating variables for the filtering of the raw disparity map.
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
lmbda = 8000
sigma = 1.5
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

start = time.time()
while True:
    # We use the .grab() method to reduce the lag between the two videos.
    if not (CamL.grab() and CamR.grab()):
        print("No more frames")
        break
    # Once we grabbed the frame, we can retreive the data from it. The .read()
    # method does everything at once, so when the data from CamL has been read,
    # the image on CamR has already changed a bit. That is why the .grad() and
    # .retreive() pair is preferable.
    _, imgL = CamL.retrieve()
    _, imgR = CamR.retrieve()

    # This piece of code rectifies the images using the precalculated fundamental matrix.
    # We could use the real-time rectification, where you create a different fundamental
    # matrix for each case, but this is an approach that works best for single images,
    # as its more computationally intensive than just applying a matrix that is defined.
    # Moreover, real-time rectification gives twitchy results, since the frames are modified
    # using a different fundamental matrix every time.
    frame_right = cv2.remap(imgR, stereoMapR_x, stereoMapR_y,
                            cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(imgL, stereoMapL_x, stereoMapL_y,
                           cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Calculate raw unfiltered disparity and show, comment out the rest if only this is wanted.
    disparity = stereo.compute(frame_left, frame_right).astype(np.float32)
    disparity = cv2.normalize(disparity, 0, 255, cv2.NORM_MINMAX)
    cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imshow("disp", disparity)  # comment out the rest.

    # displ = left_matcher.compute(frame_left, frame_right)
    # dispr = right_matcher.compute(frame_right, frsame_left)

    # filtered_disparity = wls_filter.filter(disparity_map_left = displ,left_view = frame_left, disparity_map_right = dispr, right_view = frame_right)
    # cv2.applyColorMap(filtered_disparity, cv2.COLORMAP_JET)
    # cv2.imshow("filtered_disp", deps.clamp(filtered_disparity))

    # MODIFY THIS TO YOUR HEARTS CONTENT. A useful expansion is
    # dst = cv2.addWeighted(frame_left,0.5,frame_right,0.7,0), which overlaps the two
    # rectified images, giving you a good idea of the current alignment.

    now = time.time()
    print(now - start)
    # Close window using esc key
    if cv2.waitKey(1) == 27:
        break
