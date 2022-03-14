import numpy as np
import cv2

# When using an IP Webcam application, these would be the IP adresses of the
# cell phones you are using as your webcams. The left and right phones respectively.
base_url_L = 'http://10.19.203.50:8080'
base_url_R = 'http://10.18.199.231:8080'

# Create videoCapture objects for both video streams.
CamL = cv2.VideoCapture(base_url_L+'/video')
CamR = cv2.VideoCapture(base_url_R+'/video')

num = 0

# Set infinite loop to capture images from video.
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

    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        # Put whatever directory is convenient for you.
        cv2.imwrite('data/stationary_not_board/imageL' +
                    str(num) + '.png', imgL)
        cv2.imwrite('data/stationary_not_board/imageR' +
                    str(num) + '.png', imgR)
        print("images saved!")
        num += 1
    # Displaying the capture in a single window, more convenient.
    Hori = np.concatenate((imgL, imgR), axis=1)
    cv2.imshow('Concat', Hori)

cv2.destroyAllWindows()
