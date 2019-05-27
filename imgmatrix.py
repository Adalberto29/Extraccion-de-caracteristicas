import cv2
import numpy as np

image = cv2.imread('im.jpg')
image = cv2.resize(image, (500, 500), interpolation = cv2.INTER_CUBIC)
# I just resized the image to a quarter of its original size
#image = cv2.resize(image, (0, 0), None, .25, .25)    #rem it out if u want smaller size

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Make the grey scale image have three channels
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_horizontal = np.hstack((image, grey_3_channel))


numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
cv2.waitKey()
img = grey
height, width = img.shape
GRID_X = int(height/5)
GRID_Y: int = int(width/5)
for x in range(0, width-1, GRID_X):
    for y in range(0, height-1, GRID_Y):
        matrices = img[(x*GRID_X):GRID_X, (y*GRID_Y):GRID_Y]


