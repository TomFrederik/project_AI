import cv2
import numpy as np
 
img_array = []

paths = [f'./mdn_{i}.png' for i in range(51)]

for filename in paths:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
out = cv2.VideoWriter('./mdn_dynamics.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()