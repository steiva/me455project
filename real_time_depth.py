import numpy as np 
import cv2
import matplotlib.pyplot as plt
import requests
import dependencies as deps
import time
import imutils

base_url_L = 'http://10.19.203.50:8080'
base_url_R = 'http://10.18.199.231:8080'

CamL = cv2.VideoCapture(base_url_L + '/video')
CamR = cv2.VideoCapture(base_url_R + '/video')

cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

minDisparity = 0
numDisparities = 128
blockSize = 5
disp12MaxDiff = 1
uniquenessRatio = 10
speckleWindowSize = 10
speckleRange = 12

# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )
# Calculating disparith using the StereoSGBM algorithm
left_matcher = stereo

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
lmbda = 8000
sigma = 1.5
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

start = time.time()
while True:
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()

    frame_right = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    disparity = stereo.compute(frame_left, frame_right).astype(np.float32)
    disparity = cv2.normalize(disparity,0,255,cv2.NORM_MINMAX)
    cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imshow("disp", disparity) #comment out the rest.

    # displ = left_matcher.compute(frame_left, frame_right)
    # dispr = right_matcher.compute(frame_right, frame_left)

    # filtered_disparity = wls_filter.filter(disparity_map_left = displ,left_view = frame_left, disparity_map_right = dispr, right_view = frame_right)
    # cv2.applyColorMap(filtered_disparity, cv2.COLORMAP_JET)
    # cv2.imshow("disp", deps.clamp(filtered_disparity))

    now = time.time()
    print(now - start)
    # Close window using esc key
    if cv2.waitKey(1) == 27:
        break