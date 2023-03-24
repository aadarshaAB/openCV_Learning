
# import the cv2 library
import cv2 as cv

# The function cv2.imread() is used to read an image.
img= cv.imread('test.jpg')


cv.imshow('Image', img)

# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
cv.waitKey(0)

# cv.destroyAllWindows() simply destroys all the windows we created.
cv.destroyAllWindows()


