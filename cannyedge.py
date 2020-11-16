import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('inputimg2.jpg', cv2.IMREAD_UNCHANGED)

# Noise reduction step

#grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#the remove more noice I used bilateralFilter
blurred = cv2.bilateralFilter(gray,10,200,200)
#blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

#This is the median method for lower and higher threshold.
v = np.median(blurred)
sigma = 0.33

lower_thresh = int(max(0, (1.0 - sigma) * v))
upper_thresh = int(min(255, (1.0 + sigma) * v))

#THRESH_OTSU for threshold but I will use the median.
# high_thresh, thresh_im = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# lowThresh = 0.5*high_thresh

edges = cv2.Canny(blurred,lower_thresh,upper_thresh)

plt.subplot(121),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges, cmap="gray")
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


