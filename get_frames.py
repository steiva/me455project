import numpy as np
import cv2

import numpy as np
import cv2

base_url_L = 'http://10.19.203.50:8080'
base_url_R = 'http://10.18.199.231:8080'


CamL = cv2.VideoCapture(base_url_L+'/video')
CamR = cv2.VideoCapture(base_url_R+'/video')

num = 0

while True:
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()

    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('data/stationary_not_board/imageL' + str(num) + '.png', imgL)
        cv2.imwrite('data/stationary_not_board/imageR' + str(num) + '.png', imgR)
        print("images saved!")
        num += 1
    Hori = np.concatenate((imgL, imgR), axis = 1)

    cv2.imshow('Concat',Hori)
cv2.destroyAllWindows()