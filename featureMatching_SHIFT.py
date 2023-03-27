# Brute-Force Matching with SIFT Descriptors and Ratio Test
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1=cv.imread("1Matching_BT.jpg")
img2=cv.imread("2Matching_BT.jpg")

gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create()


kp1,des1=sift.detectAndCompute(gray1,None)
kp2,des2=sift.detectAndCompute(gray2, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)


good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img3 = cv.drawMatchesKnn(gray1,kp1,gray2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()
