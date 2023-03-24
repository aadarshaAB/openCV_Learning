import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    ret,frame = cap.read()
    # Convert BGR to HSV
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    #define green color for HSV
        #lower_blue = np.array([110,50,50])
        #upper_blue = np.array([130,255,255])
        #lower_red = np.array([0,100,100])
        #upper_red = np.array([10,255,255])
    lower_green = np.array([45,100,20])
    upper_green = np.array([70,255,255])
    # Threshold the HSV image to get only green colors
    mask = cv.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()