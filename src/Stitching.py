import cv2 as cv
import random as rnd
import numpy as np
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def random_point(matches, k):
    idx = rnd.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def RANSAC(matches, loops = 10000, sample_size = 5, threshold = 0.25):
    best_num_inlier = 0
    for i in range(loops):
        samples_points = random_point(matches, sample_size)
        dx = np.array([pair[2] - pair[0] for pair in samples_points])
        dy = np.array([pair[3] - pair[1] for pair in samples_points])

        mx = np.sum(dx)/len(dx)
        my = np.sum(dy)/len(dy)

        distance = np.array([(mx + matches[i][0] - matches[i][2])**2+(my + matches[i][1] - matches[i][3])**2 for i in range(len(matches))])

        inliers_index = np.where(distance < threshold)[0]
        num_inliers = len(inliers_index)

        if num_inliers > best_num_inlier : 
            best_num_inlier = num_inliers
            best_inliers = matches[inliers_index].copy()
            best_translation = np.array([-mx, -my]).copy()

    print(best_num_inlier, '/', len(matches))
    return best_translation

def Stitch_two(left_image, right_image, shift):
    height, width, _ = np.shape(left_image)
    new_height = int(round(height + shift[1]))
    new_width = int(round(width + shift[0]))
    diff = new_width - width

    shift_matrix_l = np.float32([[1, 0, 0], [0, 1, 0]]) 
    new_image_l = cv.warpAffine(left_image, shift_matrix_l, (new_width, new_height))
    shift_matrix_r = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]) 
    new_image_r = cv.warpAffine(right_image, shift_matrix_r, (new_width, new_height))

    black_pixel = np.array([0, 0, 0])
    for i in range(new_height):
        for j in range(new_width):
            if np.array_equal(new_image_l[i][j], black_pixel) and not np.array_equal(new_image_r[i][j], black_pixel):
                new_image_l[i][j] = new_image_r[i][j]
            elif not np.array_equal(new_image_l[i][j], black_pixel) and not np.array_equal(new_image_r[i][j], black_pixel):
                '''B = (int(new_image_l[i][j][0])*(j-diff) + int(new_image_r[i][j][0])*(width-j))/(width - diff)
                G = (int(new_image_l[i][j][1])*(j-diff) + int(new_image_r[i][j][1])*(width-j))/(width - diff)
                R = (int(new_image_l[i][j][2])*(j-diff) + int(new_image_r[i][j][2])*(width-j))/(width - diff)
                new_image_l[i][j] = np.array([B, G, R])
                print(new_image_l[i][j])
                print("----------")'''
                #new_image_l[i][j] = (new_image_l[i][j] + new_image_r[i][j])/2
                pass
            else : 
                pass

    return new_image_l

def Stitch(images, translations):
    return 
