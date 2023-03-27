#Brute-Force Matching with ORB Descriptors
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1=cv.imread("1Matching_BT.jpg")
img2=cv.imread("2Matching_BT.jpg")

gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

orb=cv.ORB_create()

kp1,des1=orb.detectAndCompute(gray1,None)
kp2,des2=orb.detectAndCompute(gray2, None)

bf=cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches=bf.match(des1,des2)

matches=sorted(matches,key=lambda x: x.distance)
img3=cv.drawMatches(img1,kp1,img2,kp2,matches[:25],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()
