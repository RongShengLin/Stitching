import MSOP
import cv2
import matplotlib.pyplot as plt 

img_file = input()
img = cv2.imread(img_file)
features = MSOP.MSOP(img)
print(features.shape)