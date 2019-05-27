import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('ARTUI LOGOTIPO.jpg')
#cv2.imshow("Original", image)
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
myarray = grey.ravel()
weights = myarray/float(255)
histograma, temp1, temp2= plt.hist(myarray, 256, [0, 256], density=True)
plt.show()
coorMax = histograma.max()
print(coorMax)
cv2.waitKey(0)
