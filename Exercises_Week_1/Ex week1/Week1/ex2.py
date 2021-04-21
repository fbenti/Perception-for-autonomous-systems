import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

# More tutorials on opencv can be found:
#https://docs.opencv.org/master/d2/d96/tutorial_py_table_of_contents_imgproc.html

path = "tetris_blocks.png"

# Load the input image (whose path was supplied via command line argument) and display the image to our screen

bgr_img = cv2.imread(path)
b,g,r = cv2.split(bgr_img)       # get b,g,r
image = cv2.merge([r,g,b])
plt.imshow(image)
plt.show()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # method is used to convert an image from one color space to another
plt.imshow(gray, cmap = 'gray')
plt.show()

# Applying edge detection we can find the outlines of objects in images
edged = cv2.Canny(gray, 30, 150) # First argument is our input image. Second and third arguments are our minVal and maxVal respectively.
plt.imshow(edged, cmap='gray')
plt.show()

# Threshold the image by setting all pixel values less than 225 to 255(white; foreground) and all pixel values >= 225 to 255 (black; background), thereby segmenting the image.
# This can be tweeked so say all pixel values less than 128.
threshold = 225
threshold_value = 255
thresh = cv2.threshold(gray, threshold, threshold_value, cv2.THRESH_BINARY_INV)[1] #The first argument is the source image,
#which should be a grayscale image. The second argument is the threshold value which is used to classify the pixel values. 
#The third argument is the maximum value which is assigned to pixel values exceeding the threshold
plt.imshow(thresh, cmap='gray')
plt.show()

# Find contours (i.e., outlines) of the foreground objects in the thresholded image
# Use BINARY images. So before finding contours, apply threshold or canny edge detection.
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
# cv2.RETR_EXTERNAL = Only the eldest in every family is taken care of.
# cv2.CHAIN_APPROX_SIMPLE = Do you need all the points on the line to represent that line? 
# No, we need just two end points of that line.
cnts = imutils.grab_contours(cnts) #mandatory
output = image.copy()

# Loop over the contours
for c in cnts:
    # draw each contour on the output image with a 3px thick black outline
    cv2.drawContours(output, [c], -1, (0,0,0), 3) #-1 draws all contours
    # cv2.drawContours(output, cnts, 3, (0,0,0), 3)
plt.imshow(output)
plt.show()

# Draw the total number of contours found in purple
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10,25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 0, 155), 2)
plt.imshow(output)
plt.show()

# We apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations = 5) # second one is called structuring element or kernel which decides the nature of operation
plt.imshow(mask, cmap = 'gray')
plt.show()

# Similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations = 5)
plt.imshow(mask, cmap='gray')
plt.show()

# A typical operation we may want to apply is to take our mask and apply a bitwise AND to our input image, keeping only the masked regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask) #Calculates the per-element bit-wise conjunction of two arrays
# or an array and a scalar
# mask = 8-bit single channel array, that specifies elements of the output array to be changed.
plt.imshow(output)
plt.show()

### A)

temp = bgr_img.copy()
hsv_image = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV) # convert BGR to HSV (hue saturation value)
mask = cv2.inRange(hsv_image, (25, 190, 20), (30, 255, 255)) # white square black everything else
temp[mask==255] = 255 # ciò che in mask è bianco, fallo bianco in temp
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
plt.show()
