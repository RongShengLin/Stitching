import cv2 as cv
import random as rnd
import numpy as np
import math
import os
import sys
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_command(argv):
    y = 0
    h1 = 0
    h2 = 0
    path = ''
    tag = 1

    for i in range(len(argv)):
        if argv[i] == '-y' : 
            if i == len(argv)-1 : 
                print("loss y")
                tag = 0
            else :
                y = int(argv[i+1])
        elif argv[i] == '-p' : 
            if i == len(argv)-1 : 
                print("loss image path")
                tag = 0
            else : 
                path = argv[i+1]
        elif argv[i] == '-h1' : 
            if i == len(argv)-1 : 
                print("loss h1")
                tag = 0
            else : 
                h1 = int(argv[i+1])
        elif argv[i] == '-h2' : 
            if i == len(argv)-1 : 
                print("loss h2")
                tag = 0
            else : 
                h2 = int(argv[i+1])
            
    
    return path, y, h1, h2, tag

def Cylindrical_reprojection(image, S, focal_length):
    height, width, _ = np.shape(image)
    new_height, new_width = round(height*focal_length/S), width
    projected_image = np.zeros((new_height, new_width,3), np.uint8)
    '''print("original shape = ", (height, width))
    print("new shape = ", (new_height, new_width))'''
    
    index_map = np.mgrid[0:new_height, 0:new_width]
    index_y, index_x = index_map[0], index_map[1]
    _x = index_map[1] - new_width / 2
    _y = index_map[0] - new_height / 2
    x_ = _x#np.arctan(_x / focal_length) * S
    y_ = (_y * S) / np.sqrt(np.power(x_, 2) + focal_length ** 2)
    x = x_ + width / 2
    y = y_ + height / 2
    base_x = np.floor(x).astype('int')
    base_y = np.floor(y).astype('int')
    a = x - base_x
    b = y - base_y
    idx = np.where(0 <= base_x)
    base_x, base_y = base_x[idx], base_y[idx]
    a, b = a[idx], b[idx]
    index_x, index_y = index_x[idx], index_y[idx]

    idx = np.where(base_x <= width - 2)
    base_x, base_y = base_x[idx], base_y[idx]
    a, b = a[idx], b[idx]
    index_x, index_y = index_x[idx], index_y[idx]

    idy = np.where(0 <= base_y)
    base_x, base_y = base_x[idy], base_y[idy]
    a, b = a[idy], b[idy]
    index_x, index_y = index_x[idy], index_y[idy]

    idy = np.where(base_y <= height - 2)
    base_x, base_y = base_x[idy], base_y[idy]
    a, b = a[idy], b[idy]
    index_x, index_y = index_x[idy], index_y[idy]
    for c in range(3):
        pixel_00 = (1 - a) * (1 - b) * image[(base_y, base_x, c)]
        pixel_01 = a * (1 - b) * image[(base_y, base_x + 1, c)]
        pixel_11 = a * b * image[(base_y + 1, base_x + 1, c)]
        pixel_10 = (1 - a) * b * image[(base_y + 1, base_x, c)]
        projected_image[(index_y, index_x, c)] = pixel_00 + pixel_01 + pixel_11 + pixel_10

    return projected_image


def corp():
    path, y_, h1, h2, tag = parse_command(sys.argv)

    if tag == 0:
        return 
    
    output_image = cv.imread(path)
    height, width, _ = np.shape(output_image)
    ry = abs(height/2-y_)
    new_f = np.sqrt(np.power(ry*width/2, 2)/(np.power(height/2, 2)-np.power(ry, 2)))
    output_image = Cylindrical_reprojection(output_image, new_f, new_f)
    output_image = output_image[h1:h2, 0:width]
    file_name = 'stitch_crop.jpg'
    cv.imwrite(file_name, output_image)

corp()