import numpy as np
import cv2
from matplotlib import pyplot as plt

# imgL = cv2.imread('cones/im2.png',0)
# imgR = cv2.imread('cones/im6.png',0)
# imgL = cv2.imread('bottle_l.jpg',0)
# imgR = cv2.imread('bottle_r.jpg',0)
imgL = cv2.imread('rectified_1.png',0)
imgR = cv2.imread('rectified_2.png',0)
# stereo = cv2.StereoBM_create(numDisparities=32, blockSize=5)
# disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity)
# plt.show()

# Setting parameters for StereoSGBM algorithm
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
disp = stereo.compute(imgL, imgR).astype(np.float32)
disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)

# Displaying the disparity map
cv2.imshow("disparity",disp)
cv2.waitKey(0)