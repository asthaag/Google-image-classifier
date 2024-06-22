# Google-image-classifier

import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np

# reading images
img1 = cv2.imread("img1.jpgnoisy.jpg")
img2 = cv2.imread("img2.jpgnoisy.jpg")
img3 = cv2.imread("img3.jpgnoisy.jpg")
img4 = cv2.imread("img4.jpgnoisy.jpg")


# applying gaussian blur method

g_blur1 = cv2.GaussianBlur(img1, (5,5), 0)
g_blur2 = cv2.GaussianBlur(img2, (5,5), 0)
g_blur3 = cv2.GaussianBlur(img3, (5,5), 0)
g_blur4 = cv2.GaussianBlur(img4, (5,5), 0)


# applying median Blur method

med_blur1 = cv2.medianBlur(img1, 5)
med_blur2 = cv2.medianBlur(img2, 5)
med_blur3 = cv2.medianBlur(img3, 5)
med_blur4 = cv2.medianBlur(img4, 5)


# applying Non-Local Means Denoising algorithm

denoised1 = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
denoised2 = cv2.fastNlMeansDenoisingColored(img2, None, 10, 10, 7, 21)
denoised3 = cv2.fastNlMeansDenoisingColored(img3, None, 10, 10, 7, 21)
denoised4 = cv2.fastNlMeansDenoisingColored(img4, None, 10, 10, 7, 21)


# display result

plt.figure(figsize=(20,15))

plt.subplot(4,4,1)
plt.imshow(img1), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,2)
plt.imshow(g_blur1), plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,3)
plt.imshow(med_blur1), plt.title('Median Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,4)
plt.imshow(denoised1), plt.title('Non-Local Means'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,5)
plt.imshow(img2), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,6)
plt.imshow(g_blur2), plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,7)
plt.imshow(med_blur2), plt.title('Median Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,8)
plt.imshow(denoised2), plt.title('Non-Local Means'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,9)
plt.imshow(img3), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,10)
plt.imshow(g_blur3), plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,11)
plt.imshow(med_blur3), plt.title('Median Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,12)
plt.imshow(denoised3), plt.title('Non-Local Means'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,13)
plt.imshow(img4), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,14)
plt.imshow(g_blur4), plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,15)
plt.imshow(med_blur4), plt.title('Median Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,16)
plt.imshow(denoised4), plt.title('Non-Local Means'), plt.xticks([]), plt.yticks([])




plt.show()

# As Non-Local Means image looks the most denoised, so we will apply things on this image only

# Convert the Non-Local Means images to different color spaces (RGB, HSV, LAB)

# converting to RGB

rgb1 = cv2.cvtColor(denoised1, cv2.COLOR_BGR2RGB)
rgb2 = cv2.cvtColor(denoised2, cv2.COLOR_BGR2RGB)
rgb3 = cv2.cvtColor(denoised3, cv2.COLOR_BGR2RGB)
rgb4 = cv2.cvtColor(denoised4, cv2.COLOR_BGR2RGB)


# converting to HSV

hsv1 = cv2.cvtColor(denoised1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(denoised2, cv2.COLOR_BGR2HSV)
hsv3 = cv2.cvtColor(denoised3, cv2.COLOR_BGR2HSV)
hsv4 = cv2.cvtColor(denoised4, cv2.COLOR_BGR2HSV)


# converting to LAB

lab1 = cv2.cvtColor(denoised1, cv2.COLOR_BGR2LAB)
lab2 = cv2.cvtColor(denoised2, cv2.COLOR_BGR2LAB)
lab3 = cv2.cvtColor(denoised3, cv2.COLOR_BGR2LAB)
lab4 = cv2.cvtColor(denoised4, cv2.COLOR_BGR2LAB)


# plotting different color spaces

plt.figure(figsize=(15,15))

plt.subplot(4,4,1)
plt.imshow(denoised1), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,2)
plt.imshow(rgb1), plt.title('RGB'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,3)
plt.imshow(hsv1), plt.title('HSV'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,4)
plt.imshow(lab1), plt.title('LAB'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,5)
plt.imshow(denoised2), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,6)
plt.imshow(rgb2), plt.title('RGB'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,7)
plt.imshow(hsv2), plt.title('HSV'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,8)
plt.imshow(lab2), plt.title('LAB'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,9)
plt.imshow(denoised3), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,10)
plt.imshow(rgb3), plt.title('RGB'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,11)
plt.imshow(hsv3), plt.title('HSV'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,12)
plt.imshow(lab3), plt.title('LAB'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,13)
plt.imshow(denoised4), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,14)
plt.imshow(rgb4), plt.title('RGB'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,15)
plt.imshow(hsv4), plt.title('HSV'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,16)
plt.imshow(lab4), plt.title('LAB'), plt.xticks([]), plt.yticks([])

plt.show()

# Analyze how color channels reveal unique features (e.g., plant chlorophyll in green channel, bioluminescence in blue channel)

blue_channel1, green_channel1, red_channel1 = cv2.split(denoised1)
blue_channel2, green_channel2, red_channel2 = cv2.split(denoised2)
blue_channel3, green_channel3, red_channel3 = cv2.split(denoised3)
blue_channel4, green_channel4, red_channel4 = cv2.split(denoised4)


# plotting 

plt.figure(figsize=(15,15))

plt.subplot(4,4,1)
plt.imshow(denoised1), plt.title('Denoised Image'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,2)
plt.imshow(blue_channel1), plt.title('Blue Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,3)
plt.imshow(green_channel1), plt.title('Green Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,4)
plt.imshow(red_channel1), plt.title('Red Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,5)
plt.imshow(denoised2), plt.title('Denoised Image'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,6)
plt.imshow(blue_channel2), plt.title('Blue Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,7)
plt.imshow(green_channel2), plt.title('Green Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,8)
plt.imshow(red_channel2), plt.title('Red Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,9)
plt.imshow(denoised3), plt.title('Denoised Image'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,10)
plt.imshow(blue_channel3), plt.title('Blue Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,11)
plt.imshow(green_channel3), plt.title('Green Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,12)
plt.imshow(red_channel3), plt.title('Red Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,13)
plt.imshow(denoised4), plt.title('Denoised Image'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,14)
plt.imshow(blue_channel4), plt.title('Blue Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,15)
plt.imshow(green_channel4), plt.title('Green Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,16)
plt.imshow(red_channel4), plt.title('Red Channel'), plt.xticks([]), plt.yticks([])

plt.show()

# Create visualizations that highlight specific color components

# Converting denoised images from bgr to hsv

HSV1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
HSV2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
HSV3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
HSV4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)

# Using mask of blue color 

lower_blue1 = np.array([90,50,50])
upper_blue1 = np.array([130,255,255])
mask_blue1 = cv2.inRange(HSV1, lower_blue1, upper_blue1)
result_blue1 = cv2.bitwise_and(HSV1,HSV1, mask = mask_blue1)

lower_blue2 = np.array([90,50,50])
upper_blue2 = np.array([130,255,255])
mask_blue2 = cv2.inRange(HSV2, lower_blue2, upper_blue2)
result_blue2 = cv2.bitwise_and(HSV2,HSV2, mask = mask_blue2)

lower_blue3 = np.array([90,50,50])
upper_blue3 = np.array([130,255,255])
mask_blue3 = cv2.inRange(HSV3, lower_blue3, upper_blue3)
result_blue3 = cv2.bitwise_and(HSV3,HSV3, mask = mask_blue3)

lower_blue4 = np.array([90,50,50])
upper_blue4 = np.array([130,255,255])
mask_blue4 = cv2.inRange(HSV4, lower_blue4, upper_blue4)
result_blue4 = cv2.bitwise_and(HSV4,HSV4, mask = mask_blue4)

# plotting

plt.figure(figsize=(20,15))

plt.subplot(4,2,1)
plt.imshow(img1), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,2,2)
plt.imshow(result_blue1), plt.title('Blue Filtered'), plt.xticks([]), plt.yticks([])

plt.subplot(4,2,3)
plt.imshow(img2), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,2,4)
plt.imshow(result_blue2), plt.title('Blue Filtered'), plt.xticks([]), plt.yticks([])

plt.subplot(4,2,5)
plt.imshow(img3), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,2,6)
plt.imshow(result_blue3), plt.title('Blue Filtered'), plt.xticks([]), plt.yticks([])

plt.subplot(4,2,7)
plt.imshow(img4), plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4,2,8)
plt.imshow(result_blue4), plt.title('Blue Filtered'), plt.xticks([]), plt.yticks([])

plt.show()

 
