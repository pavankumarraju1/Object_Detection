import cv2
import imutils
import time

cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 500

while True:
    _,img=cam.read()
    img = imutils.resize(img,width=500)
    grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurImage = cv2.GaussianBlur(grayImage,(21,21),0)
    if firstFrame is None:
        firstFrame = blurImage
        continue
    
    imgdiff = cv2.absdiff(firstFrame,blurImage)
    threshImage = cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
    threshImage = cv2.dilate(threshImage,None,iterations=2)
    cnts = cv2.findContours(threshImage.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLEX)
    cnts =imutils.grab_contours(cnts)
    


    cv2.imshow("video streaming",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()