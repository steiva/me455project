# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# img1 = cv.imread('data/img1-1.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('data/img2-1.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# # Initiate SIFT detector
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# imgL = cv2.imread('cones/im2.png',0)
# imgR = cv2.imread('cones/im6.png',0)
# imgL = cv2.imread('bottle_l.jpg',0)
# imgR = cv2.imread('bottle_r.jpg',0)
imgL = cv.imread('rectified_1.png',0)
imgR = cv.imread('rectified_2.png',0)
# stereo = cv2.StereoBM_create(numDisparities=32, blockSize=5)
# disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity)
# plt.show()

# imgL = cv2.imread("../im0.png",0)
# imgR = cv2.imread("../im1.png",0)




                        # # Setting parameters for StereoSGBM algorithm
                        # minDisparity = 0
                        # numDisparities = 128
                        # blockSize = 5
                        # disp12MaxDiff = 1
                        # uniquenessRatio = 10
                        # speckleWindowSize = 10
                        # speckleRange = 12

                        # # Creating an object of StereoSGBM algorithm
                        # stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
                        #         numDisparities = numDisparities,
                        #         blockSize = blockSize,
                        #         disp12MaxDiff = disp12MaxDiff,
                        #         uniquenessRatio = uniquenessRatio,
                        #         speckleWindowSize = speckleWindowSize,
                        #         speckleRange = speckleRange
                        #     )




                        # # Calculating disparith using the StereoSGBM algorithm
                        # disp = stereo.compute(imgL, imgR).astype(np.float32)
                        # disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)

                        # # Displaying the disparity map
                        # cv2.imshow("disparity",disp)
                        # cv2.waitKey(0)

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

stereo = cv.StereoSGBM_create(
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
disparity_SGBM = stereo.compute(img1_undistorted, img2_undistorted)

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
cv.imshow("Disparity", disparity_SGBM)
cv.imwrite("disparity_SGBM_norm.png", disparity_SGBM)

# Reading the left and right images
# def stereo_match(imgL, imgR):
#     # disparity range is tuned for 'aloe' image pair
#     window_size = 15
#     min_disp = 16
#     num_disp = 96 - min_disp
#     stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
#                                    numDisparities=num_disp,
#                                    blockSize=16,
#                                    P1=8 * 3 * window_size ** 2,
#                                    P2=32 * 3 * window_size ** 2,
#                                    disp12MaxDiff=1,
#                                    uniquenessRatio=10,
#                                    speckleWindowSize=150,
#                                    speckleRange=32
#                                    )

#     # print('computing disparity...')
#     disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

#     # print('generating 3d point cloud...',)
#     h, w = imgL.shape[:2]
#     f = 0.8 * w  # guess for focal length
#     Q = np.float32([[1, 0, 0, -0.5 * w],
#                     [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
#                     [0, 0, 0, -f],  # so that y-axis looks up
#                     [0, 0, 1, 0]])
#     points = cv2.reprojectImageTo3D(disp, Q)
#     colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
#     mask = disp > disp.min()
#     out_points = points[mask]
#     out_colors = colors[mask]
#     #append_ply_array(out_points, out_colors)

#     disparity_scaled = (disp - min_disp) / num_disp
#     disparity_scaled += abs(np.amin(disparity_scaled))
#     disparity_scaled /= np.amax(disparity_scaled)
#     disparity_scaled[disparity_scaled < 0] = 0
#     return np.array(255 * disparity_scaled, np.uint8)

# arr = stereo_match(imgL, imgR)
# plt.imshow(arr)
# plt.show()

# import sys
# #import cv2
# #import numpy as np
# import time


# def find_depth(right_point, left_point, frame_right, frame_left, baseline,f, alpha):

#     # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
#     height_right, width_right, depth_right = frame_right.shape
#     height_left, width_left, depth_left = frame_left.shape

#     if width_right == width_left:
#         f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

#     else:
#         print('Left and right camera frames do not have the same pixel width')

#     x_right = right_point[0]
#     x_left = left_point[0]

#     # CALCULATE THE DISPARITY:
#     disparity = x_left-x_right      #Displacement between left and right frames [pixels]

#     # CALCULATE DEPTH z:
#     zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]

#     return zDepth