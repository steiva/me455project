import numpy as np
import cv2 as cv

# -----------------------------General Functions---------------------------------


def clamp(img):
    '''
    This function clamps an image down based on finding the second least value
    of the image and adjusting all the values less than that to the second least.
    This function is useful in the filtering case, where many values are outliers.
    :param img: input image to clamp.
    :returns: clamped image.
    '''
    arr_min = sorted(list(set(img.flatten())))[1]
    arr_max = np.max(img)
    filtered_img = np.clip(img, a_min=arr_min, a_max=arr_max)
    filtered_img = cv.normalize(
        filtered_img, filtered_img, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    filtered_img = np.uint8(filtered_img)
    return filtered_img


def disp_filtering(imgL_rect, imgR_rect, left_matcher, lmbda=8000, sigma=1.5):
    '''
    Function filters the raw disparity map, giving a better representation.
    :param imgL_rect: left rectified image.
    :param imgR_rect: right rectified image.
    :param left_matcher: a matcher object like cv.StereoSGBM_create().
    :param lmbda: Lambda is a parameter defining the amount of regularization 
    during filtering. Larger values force filtered disparity map edges to 
    adhere more to source image edges. Typical value is 8000.
    :param sigma: SigmaColor is a parameter defining how sensitive the filtering
    process is to source image edges. Large values can lead to disparity leakage
    through low-contrast edges. Small values can make the filter too sensitive to
    noise and textures in the source image. Typical values range from 0.8 to 2.0.
    :returns: filtered left and right disparity maps.
    '''
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(
        matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL_rect, imgR_rect)
    dispr = right_matcher.compute(imgR_rect, imgL_rect)
    fil_disp_left = wls_filter.filter(
        disparity_map_left=displ, left_view=imgL_rect, disparity_map_right=dispr)
    fil_disp_right = wls_filter.filter(
        disparity_map_left=displ, left_view=imgR_rect, disparity_map_right=dispr)
    return fil_disp_left, fil_disp_right

# ------------------Functions for the frame by frame method----------------------


def find_matches(img1, img2):
    '''
    Function takes in two images and finds best matches between them.
    :param img1: first image.
    :param img2: second image. 
    :returns: list with good matches, lists with coords from imgL and imgR.
    '''
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return good, pts1, pts2


def rectify_images(img1, img2, pts1, pts2):
    '''
    Rectification of images using the Fundamental Matrix.
    :param img1: first image to rectify.
    :param img2: second image to rectify.
    :param pts1: feature points in the first image.
    :param pts2: feature points in the second image.
    :returns: rectified images 1 and 2.
    '''
    fundamental_matrix, inliers = cv.findFundamentalMat(
        pts1, pts2, cv.FM_RANSAC)

    h1, w1 = img1.shape[:-1]
    h2, w2 = img2.shape[:-1]
    _, H1, H2 = cv.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
    )

    img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
    return img1_rectified, img2_rectified

# -------------Function for the precalculated fundamental matrix-----------------------


def rect_using_fmatrix(imgL, imgR, path='stereoMap.xml'):
    '''
    Function rectifies images using a predefined fundamental matrix,
    recorded in stereoMap.xml, using the rectification.py script.
    :param imgL: left image.
    :param imgR: right image.
    :param path: path to the stereoMap.xml file. 
    :returns: recitifed left and right images.
    '''
    cv_file = cv.FileStorage()
    cv_file.open(path, cv.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
    imgL_rect = cv.remap(imgL, stereoMapL_x, stereoMapL_y,
                         cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    imgR_rect = cv.remap(imgR, stereoMapR_x, stereoMapR_y,
                         cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    return imgL_rect, imgR_rect
