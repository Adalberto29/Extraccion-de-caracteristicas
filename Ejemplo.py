import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("ARTUI LOGOTIPO.jpg")
img = cv2.resize(img, (500, 500), interpolation = cv2.INTER_CUBIC)

height, width, channels = img.shape
GRID_X = int(height/5)
GRID_Y = int(width/5)
for x in range(0, width-1, GRID_X):
    cv2.line(img, (x, 0), (x, height), (255, 0, 0), 1, 1)
    for y in range(0, height-1, GRID_Y):
        cv2.line(img, (0, y), (width, y), (255, 0, 0), 1, 1)

cv2.imshow('Hehe', img)
key = cv2.waitKey(0)