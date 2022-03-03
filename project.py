import numpy as np
import cv2
import requests
import imutils

url = "http://10.19.123.167:8080/shot.jpg"
url2 = "http://10.19.189.0:8080/shot.jpg"
  
# While loop to continuously fetching data from the Url
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    
    #print(img.shape)
    #cv2.imshow("Android_cam", img)
    
    img_resp2 = requests.get(url2)
    img_arr2 = np.array(bytearray(img_resp2.content), dtype=np.uint8)
    img2 = cv2.imdecode(img_arr2, -1)
    img2 = imutils.resize(img2, width=1000, height=1800)
    #img2 = cv2.rotate(img2, cv2.ROTATE_180)
    #print(img2.shape)

    Hori = np.concatenate((img, img2), axis = 1)
    cv2.imshow("Android_cam", Hori)
    cv2.imwrite("image_concat-6.jpg", Hori)
    cv2.imwrite("img1-6.jpg", img)
    cv2.imwrite("img2-6.jpg", img2)
    break
    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break
        
cv2.destroyAllWindows()