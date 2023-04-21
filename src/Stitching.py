import cv2 as cv
import random as rnd
import numpy as np
import math
import os
import sys
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import test_lib

def parse_command(argv):
    mask = -1
    order = 1
    path = ''
    tag = 1

    for i in range(len(argv)):
        if argv[i] == '-o' : 
            if i == len(argv)-1 : 
                print("loss order type")
                tag = 0
            elif argv[i+1] != '-1' : 
                print("wrong order")
                tag = 0
            else :
                order = -1
        elif argv[i] == '-p' : 
            if i == len(argv)-1 : 
                print("loss image path")
                tag = 0
            else : 
                path = argv[i+1]
        elif argv[i] == '-s' : 
            if i == len(argv)-1 : 
                print("loss mask size")
                tag = 0
            else : 
                mask = int(argv[i+1])
                if mask <= 0: 
                    print("mask size can't be negative")
                    tag = 0
                elif mask >= 20:
                    print("maximum mask size is 20")
                    tag = 0
    
    return path, order, mask, tag

def LoadImages(path):
    images = []
    focal_length = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        img = cv.imread(os.path.join(path, tokens[0]))
        images.append(img)
        focal_length.append(float(tokens[1]))
    return images, focal_length

def Cylindrical_projection(image, S, focal_length):
    height, width, _ = np.shape(image)
    new_height, new_width = round(height*S/focal_length), round(math.atan(width/(focal_length*2))*S*2)
    projected_image = np.zeros((new_height, new_width,3), np.uint8)
    '''print("original shape = ", (height, width))
    print("new shape = ", (new_height, new_width))'''

    for i in range(new_height):
        for j in range(new_width):
            _x = j - new_width/2
            _y = i - new_height/2
            x_ = math.tan(_x/S)*focal_length
            y_ = (_y/S)*math.sqrt(x_**2+focal_length**2)
            x = x_ + width/2
            y = y_ + height/2
            base_x = int(np.floor(x))
            base_y = int(np.floor(y))

            if(base_x < 0 or base_x > width-2 or base_y < 0 or base_y > height-2) : continue
            
            a = x - base_x
            b = y - base_y
            pixel_00 = (1-a)*(1-b)*image[base_y][base_x]
            pixel_01 = a*(1-b)*image[base_y][base_x+1]
            pixel_11 = a*b*image[base_y+1][base_x+1]
            pixel_10 = (1-a)*b*image[base_y+1][base_x]
            projected_image[i][j] = pixel_00 + pixel_01 + pixel_11 + pixel_10

    return projected_image

def Random_point(matches, k):
    idx = rnd.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def RANSAC(matches, loops = 10000, sample_size = 5, threshold = 0.25):
    best_num_inlier = 0
    for i in range(loops):
        samples_points = Random_point(matches, sample_size)
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
            best_translation = np.array([round(-mx), round(-my)]).copy()

    #print(best_num_inlier, '/', len(matches))
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
                B = (int(new_image_l[i][j][0])*(width-j) + int(new_image_r[i][j][0])*(j-diff))/(width - diff)
                G = (int(new_image_l[i][j][1])*(width-j) + int(new_image_r[i][j][1])*(j-diff))/(width - diff)
                R = (int(new_image_l[i][j][2])*(width-j) + int(new_image_r[i][j][2])*(j-diff))/(width - diff)
                new_image_l[i][j] = np.array([B, G, R])
                '''print(new_image_l[i][j])
                print("----------")'''
                #new_image_l[i][j] = (new_image_l[i][j] + new_image_r[i][j])/2
                #new_image_l[i][j] = new_image_r[i][j]
                #pass
            else : 
                pass

    return new_image_l

def Stitch():
    path, order, mask_size, tag = parse_command(sys.argv)
    if tag == 0 : 
        return 
    
    images, focal_lengths = LoadImages(path)
    if len(images) <= 1 : 
        print("number of images isn't enough")
        return 
    
    # default clockwise
    if order == -1: 
        images.reverse()
        focal_lengths.reverse()

    # Cylindrical_projection
    avg_focal_length = sum(focal_lengths)/len(focal_lengths)

    for i in range(len(images)):
        images[i] = Cylindrical_projection(images[i], avg_focal_length, focal_lengths[i])

    # calculate local shifts and output image size
    images_shifts = [np.array([0, 0])]
    x_shift = 0
    y_shift = 0
    y_shift_p = 0
    y_shift_n = 0
    for i in range(len(images)-1):
        left_gray_image = cv.cvtColor(images[i], cv.COLOR_RGB2GRAY)
        right_gray_image = cv.cvtColor(images[i+1], cv.COLOR_RGB2GRAY)
        kp_left, des_left = test_lib.OpenCV_SIFT(left_gray_image)
        kp_right, des_right = test_lib.OpenCV_SIFT(right_gray_image)
        matches = test_lib.OpenCV_matcher(kp_left, des_left, cv.cvtColor(images[i], cv.COLOR_BGR2RGB), kp_right, des_right, cv.cvtColor(images[i-1], cv.COLOR_BGR2RGB), 0.5)
        shift = RANSAC(matches)
        x_shift = x_shift + shift[0]
        y_shift = y_shift + shift[1]
        y_shift_p = max(y_shift_p, y_shift)
        y_shift_n = min(y_shift_n, y_shift)
        images_shifts.append(np.array(shift))

    new_height, new_width, _ = np.shape(images[0])
    new_height = round(new_height + (y_shift_p - y_shift_n))
    new_width = round(new_width + x_shift)

    print("new_height =", new_height, "new_width =", new_width)

    # calculate global shift
    images_shifts[0][1] = y_shift_n*-1
    accumulate_shift = np.array([0, 0])
    r_offset = []
    for i in range(len(images_shifts)): 
        accumulate_shift = images_shifts[i] = images_shifts[i] + accumulate_shift
        shift_matrix = np.float32([[1, 0, images_shifts[i][0]], [0, 1, images_shifts[i][1]]])
        r_offset.append(images_shifts[i][0] + np.shape(images[i])[1])
        images[i] = cv.warpAffine(images[i], shift_matrix, (new_width, new_height))
        '''ldr1 = images[i][:,:,::-1]
        plt.imshow(ldr1)
        plt.show()'''

    # stitch all image
    output_image = np.zeros((new_height, new_width,3), np.uint8)
    black_pixel = np.array([0, 0, 0])
    index = 0
    r = r_offset[index]
    l = np.floor(images_shifts[index+1][0])
    mid = round((np.floor(images_shifts[index+1][0])+r)/2)
    m_l = mid - mask_size
    m_r = mid + mask_size
    for j in range(new_width):
        if j >= r : 
            if (index+1) < (len(images)-1):
                index += 1
                r = r_offset[index]
                l = np.floor(images_shifts[index+1][0])
                mid = round((np.floor(images_shifts[index+1][0])+r)/2)
                m_l = mid - mask_size
                m_r = mid + mask_size
            else : 
                r = new_width
        
        for i in range(new_height):
            L_pixel = images[index][i][j]
            R_pixel = images[index+1][i][j]

            if mask_size == -1:
                if not np.array_equal(L_pixel, black_pixel) and np.array_equal(R_pixel, black_pixel):
                    output_image[i][j] = L_pixel  
                elif not np.array_equal(L_pixel, black_pixel) and not np.array_equal(R_pixel, black_pixel):
                    B = (int(L_pixel[0])*(r-j) + int(R_pixel[0])*(j-l))/(r - l)
                    G = (int(L_pixel[1])*(r-j) + int(R_pixel[1])*(j-l))/(r - l)
                    R = (int(L_pixel[2])*(r-j) + int(R_pixel[2])*(j-l))/(r - l)
                    output_image[i][j] = np.array([B, G, R])
                elif np.array_equal(L_pixel, black_pixel) and not np.array_equal(R_pixel, black_pixel):
                    output_image[i][j] = R_pixel
            else :
                if j < m_l : 
                    output_image[i][j] = L_pixel
                elif j > m_r : 
                    output_image[i][j] = R_pixel
                else :
                    if not np.array_equal(L_pixel, black_pixel) and np.array_equal(R_pixel, black_pixel):
                        output_image[i][j] = L_pixel  
                    elif not np.array_equal(L_pixel, black_pixel) and not np.array_equal(R_pixel, black_pixel):
                        B = (int(L_pixel[0])*(m_r-j) + int(R_pixel[0])*(j-m_l))/(m_r - m_l)
                        G = (int(L_pixel[1])*(m_r-j) + int(R_pixel[1])*(j-m_l))/(m_r - m_l)
                        R = (int(L_pixel[2])*(m_r-j) + int(R_pixel[2])*(j-m_l))/(m_r - m_l)
                        if B == G and G == R and R == 0 : 
                            print(L_pixel, R_pixel)
                            print((m_r-j), (j-m_l), (m_r - m_l))
                        output_image[i][j] = np.array([B, G, R])
                    elif np.array_equal(L_pixel, black_pixel) and not np.array_equal(R_pixel, black_pixel):
                        output_image[i][j] = R_pixel
                    

        
    test = output_image[:,:,::-1]
    plt.imshow(test)
    plt.show()
    cv.imwrite('stitch_output.jpg', output_image)
    return output_image

def test():
    image_path = sys.argv[1]
    images, focal_lengths = LoadImages(image_path)
    left_image = Cylindrical_projection(images[1], focal_lengths[1])
    right_image = Cylindrical_projection(images[0], focal_lengths[0])
    left_gray_image = cv.cvtColor(left_image, cv.COLOR_RGB2GRAY)
    right_gray_image = cv.cvtColor(right_image, cv.COLOR_RGB2GRAY)

    kp_left, des_left = test_lib.OpenCV_SIFT(left_gray_image)
    kp_right, des_right = test_lib.OpenCV_SIFT(right_gray_image)
    matches = test_lib.OpenCV_matcher(kp_left, des_left, cv.cvtColor(left_image, cv.COLOR_BGR2RGB), kp_right, des_right, cv.cvtColor(right_image, cv.COLOR_BGR2RGB), 0.5)
    translation = RANSAC(matches)
    
    print(translation)
    new_image = Stitch_two(left_image, right_image, translation)
    ldr1 = new_image[:,:,::-1]
    plt.imshow(ldr1)
    plt.show()
    return 

#test()
Stitch()
