import cv2
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from matplotlib.patches import Rectangle

ratio = [0.06, 0.67, 0.27]

def find_possible_feature(F, height, width, n, E, Eliminating_edge = True):
    #find 3 * 3 max and above 10
    possible_feature = []
    three_to_three = [F[0:height-2, 0:width-2], F[0:height-2, 1:width-1], F[0:height-2, 2:width],
                     F[1:height-1, 0:width-2], F[1:height-1, 1:width-1], F[1:height-1, 2:width],
                     F[2:height, 0:width-2], F[2:height, 1:width-1], F[2:height, 2:width]]
    max_F = np.amax(np.transpose(three_to_three, (1, 2, 0)), axis=2)
    center_F = F[1:height-1, 1:width-1]
    larger_than_ten = np.where(center_F > 10, center_F, -1)
    valid_pos = np.where(max_F == larger_than_ten)
    x = valid_pos[0]
    y = valid_pos[1]
    for k in range(x.shape[0]):
        i, j = int(x[k]) + 1, int(y[k]) + 1
        if not Eliminating_edge or E[i][j] < (11 ** 2) / 10:
            origin = [i, j, n]
            possible_feature.append([i, j, F[i, j], origin])
    
    if n == 0:
        return possible_feature
    else:
        sub_possible = []
        for x, y, f, s in possible_feature:
            origin = [x, y, n]
            df_dx = (F[x + 1, y] - F[x - 1, y]) / 2
            df_dy = (F[x, y + 1] - F[x, y - 1]) / 2
            df_dxdx = F[x + 1, y] - 2 * F[x, y] + F[x - 1, y]
            df_dydy = F[x, y + 1] - 2 * F[x, y] + F[x, y - 1]
            df_dxdy = (F[x - 1, y - 1] - F[x - 1, y + 1] + F[x + 1, y + 1] - F[x + 1, y - 1]) / 4
            if (df_dxdx * df_dydy - df_dxdy * df_dxdy) == 0:
                sub_possible.append([x * pow(2, n), y * pow(2, n), F[x, y], origin])
            else:
                A = np.array([[df_dxdx, df_dxdy], [df_dxdy, df_dydy]])
                A_I = linalg.inv(A)
                x_m = - np.matmul(A_I, np.array([df_dx, df_dy]))
                #print(x_m)
                F_r = f + np.matmul(np.transpose([df_dx, df_dy]), x_m) + np.matmul(np.transpose(x_m), np.matmul(A, x_m)) / 2
                x_r, y_r = round((x + x_m[0]) * pow(2, n)), round((y + x_m[1])  * pow(2, n))
                if 0 <= x_r and x_r < height * pow(2, n) and 0 <= y_r and y_r < width * pow(2, n):
                    sub_possible.append([x_r , y_r, F_r, origin])
        return sub_possible

def calculate_H(G, height, width):
    H = np.zeros((height, width, 2, 2))
    H[:, :, 0, 0] = G[:, :, 0] * G[:, :, 0]
    H[:, :, 0, 1] = G[:, :, 0] * G[:, :, 1]
    H[:, :, 1, 0] = G[:, :, 1] * G[:, :, 0]
    H[:, :, 1, 1] = G[:, :, 1] * G[:, :, 1]
    for i in range(2):
        for j in range(2):
            H[:, :, i, j] = cv2.GaussianBlur(H[:, :, i, j], (3, 3), 1.5)
    return H

def calculate_gradients(P):
    G = np.zeros((P.shape[0], P.shape[1], 2))
    for i in range(1, P.shape[0] - 1):
        G[i, :, 0] = (P[i + 1, :] - P[i - 1, :]) / 2
    G[0, :, 0] = P[1, :] - P[0, :]
    G[-1, :, 0] = P[-1, :] - P[-2, :]
    for j in range(1, P.shape[1] - 1):
        G[:, j, 1] = (P[:, j + 1] - P[:, j - 1]) / 2
    G[:, 0, 1] = P[:, 1] - P[:, 0]
    G[:, -1, 1] = P[:, -1] - P[:, -2]
    return G

def non_maximal_suppression(possible_feature, n = 500, robust = 0.9, ratio = 1):
    possible_feature.sort(key = lambda x: x[2], reverse = True)
    
    value = np.zeros(len(possible_feature))
    x_ax = np.zeros(len(possible_feature))
    y_ax = np.zeros(len(possible_feature))
    for i in range(len(possible_feature)):
        x_ax[i] = possible_feature[i][0]
        y_ax[i] = possible_feature[i][1]
        value[i] = possible_feature[i][2]
    
    radius = []
    first_dot = possible_feature[0]
    radius.append([first_dot[0], first_dot[1], float('inf'), first_dot[3]])
    for i in range(1, round(len(possible_feature) * ratio)):
        n_value = value[:i] * robust
        neighbor = np.where(n_value > value[i])
        if neighbor[0].shape[0] == 0:
            radius.append([possible_feature[i][0], possible_feature[i][1], float('inf'), possible_feature[i][3]])
            continue
        r_min = float('inf')
        r = np.sqrt((x_ax[i] - x_ax[neighbor]) ** 2 + (y_ax[i] - y_ax[neighbor]) ** 2)
        r_min = np.min(r)
        radius.append([possible_feature[i][0], possible_feature[i][1], r_min, possible_feature[i][3]])
    return radius

def rectan(x, y, theta):
    x1, y1 = round(x - 20 * (abs(theta[0]) + abs(theta[1]))) - 5, round(y - 20 * (abs(theta[0]) + abs(theta[1]))) - 5
    x2, y2 = 5 + round(x + 20 * (abs(theta[0]) + abs(theta[1]))), 5 + round(y + 20 * (abs(theta[0]) + abs(theta[1])))
    #print(x, y, x1, y1, x2, y2)
    return x1, y1, x2, y2

def feature_descriptor(x, y, s, P_l, select_features, x_to_scale, y_to_scale):
    #Orientation assignment
    P_l_G = cv2.GaussianBlur(P_l[x - 1 :x + 2, y - 1 : y + 2], (3, 3), 4.5)
    u_l = [(P_l_G[2, 1] - P_l_G[0, 1]) / 2, (P_l_G[1, 2] - P_l_G[1, 0]) / 2]
    d_u_l = math.sqrt(u_l[0] ** 2 + u_l[1] ** 2)
    theta = [u_l[0] / d_u_l, u_l[1] / d_u_l]

    #Rotation
    height = P_l.shape[0]
    width = P_l.shape[1]
    #print(height, width)
    x1, y1, x2, y2 = rectan(x, y, theta)
    if 0 > x1 or 0 > y1 or x2 >= height or y2 >= width:
        return

    #find rotation matrix
    M = cv2.getRotationMatrix2D((x - x1, y - y1), - math.atan2(theta[0], theta[1]) * 180 / math.pi, scale=1)
    #print("theta = ", math.atan2(theta[0], theta[1]) * 180 /math.pi)

    #fig, (ax1, ax2, ax3) = plt.subplots(3)
    #ax1.imshow(P_l[x1 : x2 + 1, y1 : y2 + 1])
    #ax1.plot(y - y1, x - x1, 'r+')
    
    #rotation and sample 40 * 40 pixel
    P_r = cv2.warpAffine(P_l[x1 : x2 + 1, y1 : y2 + 1], M, (70, 70))
    P_r = P_r[x - x1 - 20 : x - x1 + 20, y - y1 - 20 : y - y1 + 20]
    
    #ax2.imshow(P_r)
    #ax2.plot(19.5, 19.5, 'r+')
    
    P_r_G = cv2.GaussianBlur(P_r, (3, 3), 2)
    #rescale
    P_five_scale = cv2.resize(P_r_G, (8, 8))
    
    #ax3.imshow(P_five_scale)
    #plt.show()
    
    u = np.mean(P_five_scale)
    dev = math.sqrt(np.var(P_five_scale))
    I = (P_five_scale - u) / dev
    feature = [x_to_scale, y_to_scale, s, theta[0], theta[1]]
    feature.extend(I.flatten().tolist())
    #print(feature)
    select_features.append(np.array(feature))
    
    ax = plt.gca()
    rect_x = y_to_scale + 20 * pow(2, s) * (theta[0] - theta[1])
    rect_y = x_to_scale - 20 * pow(2, s) * (theta[0] + theta[1])
    rect = Rectangle((rect_x, rect_y), 40 * pow(2, s), 40 * pow(2, s), math.atan2(theta[0], theta[1]) * 180 / math.pi, edgecolor = 'r', fill = False)
    ax.add_patch(rect)

    ax.plot(y_to_scale, x_to_scale, 'r+')
    ax.arrow(y_to_scale, x_to_scale, 20 * pow(2, s) * theta[1], 20 * pow(2, s) * theta[0], color = 'r')
    #plt.show()
    #exit(0)
    return 


def MSOP(img, num_of_feature = 500, scale = 5):
    img_I = ratio[0] * img[:, :, 0] + ratio[1] * img[:, :, 1] + ratio[2] * img[:, :, 2]
    #Guassain blur and rescale
    print(img_I.shape)
    P_list = []
    P_list.append(img_I)
    for i in range(1, scale):
        P_i = cv2.GaussianBlur(P_list[i - 1], (3, 3), 1)
        P_i_dot = cv2.resize(P_i, (P_i.shape[1] // 2, P_i.shape[0] // 2))
        P_list.append(P_i_dot)

    n = 0
    possible_feature = []

    for P in P_list:
        
        height = P.shape[0]
        width = P.shape[1]
        #calculate gradients
        G = calculate_gradients(P)
        
        #cv2.imwrite(f'./{n}_x.jpg', G[:, :, 0])
        #cv2.imwrite(f'./{n}_y.jpg', G[:, :, 1])

        #calculate H
        H = calculate_H(G, height, width)

        #calculate f
        F = np.zeros((height, width))
        E = np.zeros((height, width))
        delta = 1e-9
        Tr_H = H[:, :, 0, 0] + H[:, :, 1, 1]
        Det_H = H[:, :, 0, 0] * H[:, :, 1, 1] - H[:, :, 1, 0] * H[:, :, 0, 1]
        F[:, :] = Det_H / (Tr_H + delta)
        E[:, :] = (Tr_H ** 2) / (Det_H + delta)                                                                                       
        #find possible feature
        possible_feature.extend(find_possible_feature(F, height, width, n, E, Eliminating_edge = True))
        n += 1

    #Non_maximal_suppression
    radius = non_maximal_suppression(possible_feature, n = num_of_feature)
    radius.sort(key = lambda x: x[2], reverse = True)
    #i = 0
    #plt.imshow(img)
    #while i != num_of_feature and i < len(radius):
    #    plt.plot(possible_feature[i][1], possible_feature[i][0], 'r+')
    #    print(l[2])
    #    i += 1
    #plt.show()
    #print(radius)
    
    plt.imshow(img)
    i = 0
    select_feature = []
    while i != num_of_feature and i < len(radius):
        #plt.plot(radius[i][1], radius[i][0], 'r+')
        #print(l[2])
        #descriptor
        x, y = radius[i][0], radius[i][1]
        x_o, y_o, s = radius[i][3]
        feature_descriptor(x_o, y_o, s, P_list[s], select_feature, x, y)
        i += 1
    plt.show()
    #print(select_feature)
    return np.array(select_feature)

#simple matching
def simple_matching(f1, f2):
    return linalg.norm(f1 - f2)