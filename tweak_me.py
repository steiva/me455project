import numpy as np
import cv2
import matplotlib.pyplot as plt

# Input your rectified images here, and have some fun.
imgL = cv2.imread('L_rect.png', 0)
imgR = cv2.imread('R_rect.png', 0)


def nothing(x):
    pass


cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)
# Creating sliders for your parameters. Disable and enable them
# according to your preference.
cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 0, 25, nothing)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

while True:
    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities', 'disp')*16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp')*2+1
    preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp')*2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Calculating disparity using the StereoBM algorithm

    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it
    # is essential to convert it to CV_32F and scale it down 16 times.
    disp = stereo.compute(imgL, imgR).astype(np.float32)
    disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
    # Converting to float32

    # Displaying the disparity map
    cv2.imshow("disp", disp)

    # Close window using esc key
    if cv2.waitKey(1) == 27:
        break
