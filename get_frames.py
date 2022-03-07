import numpy as np
import cv2
import requests
import imutils

import numpy as np
import cv2

import imutils

url = "http://10.19.58.216:8080/shot.jpg"
url2 = "http://10.18.178.59:8080/shot.jpg"

num = 0

while True:

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)

    img_resp2 = requests.get(url2)
    img_arr2 = np.array(bytearray(img_resp2.content), dtype=np.uint8)
    img2 = cv2.imdecode(img_arr2, -1)
    img2 = imutils.resize(img2, width=1000, height=1800)
    #img2 = cv2.rotate(img2, cv2.ROTATE_180)

    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('data/not_board/image_newL' + str(num) + '.png', img)
        cv2.imwrite('data/not_board/image_newR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1
    Hori = np.concatenate((img, img2), axis = 1)

    cv2.imshow('Concat',Hori)
cv2.destroyAllWindows()