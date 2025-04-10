import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)

img = cv2.imread('image.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w,c = img.shape   

#if image has width greater than 1000 reduce it so save on time complexity
if w > 1000:
    new_w = 1000
    aspectRatio = w/h
    new_h = int(new_w/aspectRatio)
    #need to use Inter_Area interpolation when shrinking the size of an image
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Need to use otsu instead of binary because binary won't pick up on lighter colors all the time
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_OTSU)
    thresh = (255 - thresh)
    return thresh

thresh_img = thresholding(img)

#dilation
kernel = np.ones((2,85), np.int8)
dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
cv2.imshow('image', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()

