import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import dependencies as deps

# This file is an alternative for the sadbox.ipynb, for the more rigid in mind.

# importing images
imgL = cv.imread('data/not_board/imageL3.png')
imgR = cv.imread('data/not_board/imageR3.png')
# imgL = cv.imread('data/initial/img1-5.jpg')
# imgR = cv.imread('data/initial/img2-5.jpg')
# imgL = cv.imread('bottle_l.jpg')
# imgR = cv.imread('bottle_r.jpg')

# Pipeline to rectify images without using the precalculated fundamental matrix,
# getting it by matching features of the images every time and calculating it.
good, pts1, pts2 = deps.find_matches(imgL, imgR)
imgL_rect, imgR_rect = deps.rectify_images(imgL, imgR, pts1, pts2)

# Get the rectified images using the precalculated fundamental matrix from the
# chessboard method.
imgL_rect_pred, imgR_rect_pred = deps.rect_using_fmatrix(imgL, imgR)

block_size = 11
min_disp = -128
max_disp = 128
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 5
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 200
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
disp12MaxDiff = 0

matcher = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)
# Disparity for the images rectified by the frame by frame method.
disparity = matcher.compute(imgL_rect, imgR_rect)
# Disparity for the images rectified with the precalculated fundamental matrix.
disparity_pred = matcher.compute(imgL_rect_pred, imgR_rect_pred)

# Filtering the frame by frame disparity
filtered_left, filtered_right = deps.disp_filtering(
    imgL_rect, imgR_rect, matcher)
# filtering the precalculated ones.
filtered_left_pred, filtered_right_pred = deps.disp_filtering(
    imgL_rect_pred, imgR_rect_pred, matcher)

fig = plt.figure(4, figsize=(7, 7), dpi=250)
fig.patch.set_facecolor('w')

nrows = 3
ncols = 2

plt.subplot(nrows, ncols, 1)
plt.title("Left image rect.", fontsize=7)
plt.imshow(imgL_rect)

plt.subplot(nrows, ncols, 3)
plt.title("Left image disparity unfiltered.", fontsize=7)
plt.imshow(disparity)

plt.subplot(nrows, ncols, 5)
plt.title("Left image disparity map filtered.", fontsize=7)
plt.imshow(deps.clamp(filtered_left))

plt.subplot(nrows, ncols, 2)
plt.title("Right image rect.", fontsize=7)
plt.imshow(imgL_rect_pred)

plt.subplot(nrows, ncols, 4)
plt.title("Right image disparity map.", fontsize=7)
plt.imshow(disparity_pred)

plt.subplot(nrows, ncols, 6)
plt.title("Right image disparity map filtered.", fontsize=7)
plt.imshow(deps.clamp(filtered_left_pred))

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.tight_layout(pad=0)
plt.subplots_adjust(wspace=0.025, hspace=0.1)

plt.show()
