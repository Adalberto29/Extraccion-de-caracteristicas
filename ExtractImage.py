import numpy as np
import cv2
import matplotlib.pyplot as plt
# Load an color image in grayscale
img = cv2.imread('ARTUI LOGOTIPO.jpg')
im_res = cv2.resize(img, (500, 500), interpolation = cv2.INTER_CUBIC)

color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([im_res],[i],None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])
plt.hist(img.ravel(),256,[0,256]); plt.show()
plt.show()

cv2.imshow('Mi imagen', im_res)
cv2.waitKey(0)
cv2.destroyAllWindows()

