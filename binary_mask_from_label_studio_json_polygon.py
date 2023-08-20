import json

import numpy as np

import cv2



 

with open("annotations.json", "r") as read_file:

    data = json.load(read_file)

 

points = np.array(data[0]['annotations'][0]["result"][0]['value']['points'])

#The units the x, y, width and height of image annotations are provided in percentages of overall image dimension

original_width = data[0]['annotations'][0]["result"][0]['original_width']

original_height = data[0]['annotations'][0]["result"][0]['original_height']

x_points = (points[:,0] / 100) * original_width

y_points = (points[:,1] / 100 ) * original_height

contours = np.stack((x_points, y_points), axis=1).round().astype(int)

mask = np.zeros((original_height,original_width))

img3=cv2.drawContours(mask, [contours], -1, 255, -1)

print(np.count_nonzero(mask))

cv2.imwrite('binary_mask.png',mask)