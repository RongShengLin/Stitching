import cv2 as cv
import random as rnd
import numpy as np
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def Harris_corner_detection(image, show_corner = 0, K = 0.04, w_size = 3, threshold = 0.01):
    height, weight = image.shape[:2]
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = cv.GaussianBlur(gray_image, (w_size, w_size), 0)
    

    dx = cv.Sobel(gray_image, cv.CV_32F, 1, 0)
    dy = cv.Sobel(gray_image, cv.CV_32F, 0, 1)
 
    Ix2 = dx**2
    Iy2 = dy**2
    Ixy = dx*dy

    Sx = cv.GaussianBlur(Ix2, (w_size, w_size), 2)
    Sy = cv.GaussianBlur(Iy2, (w_size, w_size), 2)
    Sxy = cv.GaussianBlur(Ixy, (w_size, w_size), 2)

    R = (Sx*Sy-Sxy**2)-K*(Sx+Sy)**2

    if show_corner == 1 :
        corner_image = image.copy()
    
    output = []
    R_max = np.max(R)
    for i in range(height):
        for j in range(weight):
            local = [max(0,i-1), min(i+2,height-1), max(0,j-1), min(j+2,weight-1)]
            if R[i][j] > R_max*threshold and R[i][j] == np.max(R[local[0]:local[1], local[2]:local[3]]):
                if show_corner == 1 : corner_image[i][j][2] = 255
                output.append([i, j])

    if show_corner == 1 : return output, corner_image
    else : return output

    
def test():
    img = cv.imread('./parrington/prtn00.jpg')
    corners, img = Harris_corner_detection(img, show_corner=1)
    print(len(corners))
    ldr1 = img[:,:,::-1]
    plt.imshow(ldr1)
    plt.show()
