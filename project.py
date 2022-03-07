import numpy as np
import cv2
import requests
import imutils

# url = "http://10.19.123.167:8080/shot.jpg"
# url2 = "http://10.19.189.0:8080/shot.jpg"
# #url = "http://10.18.178.59:8080/shot.jpg"

# url = "http://10.19.58.216:8080/shot.jpg"
# url2 = "http://10.18.178.59:8080/shot.jpg"
  
# # While loop to continuously fetching data from the Url
# while True:
#     img_resp = requests.get(url)
#     img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#     img = cv2.imdecode(img_arr, -1)
#     img = imutils.resize(img, width=1000, height=1800)
    
#     #print(img.shape)
#     #cv2.imshow("Android_cam", img)
    
#     img_resp2 = requests.get(url2)
#     img_arr2 = np.array(bytearray(img_resp2.content), dtype=np.uint8)
#     img2 = cv2.imdecode(img_arr2, -1)
#     img2 = imutils.resize(img2, width=1000, height=1800)
#     img2 = cv2.rotate(img2, cv2.ROTATE_180)
#     print(img2.shape)

#     Hori = np.concatenate((img, img2), axis = 1)
#     # cv2.imshow("Android_cam", Hori)
#     # cv2.imwrite("image_concat-6.jpg", Hori)
#     cv2.imwrite("imgL.jpg", img)
#     cv2.imwrite("imgR.jpg", img2)
#     break
#     # Press Esc key to exit
#     if cv2.waitKey(1) == 27:
#         break
        
# cv2.destroyAllWindows()

import numpy as np
import cv2

import imutils

  
# cap = cv2.VideoCapture('http://10.19.58.216:8080')
# cap2 = cv2.VideoCapture('http://10.18.178.59:8080')

url = "http://10.19.58.216:8080/shot.jpg"
url2 = "http://10.18.178.59:8080/shot.jpg"

num = 0

#while cap.isOpened():
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
    

    # succes1, img = cap.read()
    # succes2, img2 = cap2.read()

    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('data/not board/imageL' + str(num) + '.png', img)
        cv2.imwrite('data/not board/imageR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1
    Hori = np.concatenate((img, img2), axis = 1)

    cv2.imshow('Concat',Hori)
    #cv2.imshow('Img 2',img2)
cv2.destroyAllWindows()

# Release and destroy all windows before termination
# cap.release()
# cap2.release()