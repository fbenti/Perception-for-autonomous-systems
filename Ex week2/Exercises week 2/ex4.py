# Optical Flow
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the images "OF1.jpg" and "OF2.jpg" and change them to grayscale.
img1 = cv2.imread('OF1.jpg')
img2 = cv2.imread('OF2.jpg')

b,g,r = cv2.split(img1) # Changing the order from bgr to rgb so that matplotlib can show it
img1 = cv2.merge([r,g,b])
b,g,r = cv2.split(img2) # Changing the order from bgr to rgb so that matplotlib can show it
img2 = cv2.merge([r,g,b])

gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# plt.figure(figsize = (18,18))
# plt.subplot(1,2,1)
# plt.imshow(gray1, cmap = 'gray')
# plt.subplot(1,2,2)
# plt.imshow(gray2, cmap = 'gray')

# They look pretty much the same. We can now use optical flow to find out how the objects 
# in the picutures have moved. We do this by first using [cv2.goodFeaturesToTrack]
# (https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541) 
# to find features in the first image. You can play around with the parameters to see the difference.
feat1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.4, minDistance=7)
# The function finds the most prominent corners in the image or in the specified image region

# Next we use the function [cv2.calcOpticalFlowPyrLK]
# (https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323) 
# to track the features in the next image.
feat2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat1, None)

# We now have the location of the features from the first image in the second image. To find the movement we can draw a line between the keypoints.
# for i in range(len(feat1)):
#     cv2.line(img2, (feat1[i][0][0], feat1[i][0][1]), (feat2[i][0][0], feat2[i][0][1]), (0, 255, 0), 2)
#     cv2.circle(img2, (feat1[i][0][0], feat1[i][0][1]), 5, (0, 255, 0), -1)

# plt.figure(figsize=(15,15))
# plt.imshow(img2)


# ## 4a) Change the code such that only the keypoints that have moved will be showed in the image.

# # feat1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.4, minDistance=7)
# # feat2 = cv2.goodFeaturesToTrack(gray2, maxCorners=100, qualityLevel=0.4, minDistance=7)
# # for i in range(len(feat1)):
# #     if feat1[i][0][0] != feat2[i][0][0] and feat1[i][0][1] != feat2[i][0][1]:
# #         cv2.line(img2, (feat1[i][0][0], feat1[i][0][1]), (feat2[i][0][0], feat2[i][0][1]), (0, 255, 0), 2)
# #         cv2.circle(img2, (feat1[i][0][0], feat1[i][0][1]), 5, (0, 255, 0), -1)
# # plt.figure(figsize=(15,15))
# # plt.imshow(img2)

# plt.show()

### Dense Optical Flow
# The sparse optical flow finds the flow of the detected keypoints.
#  We will now try to use dense optical flow which, finds the flow of all the points in the picture. 
# For this example we use the same two images, so we don't have to load them again. 
# To find the optical flow we use the function [cv2.calcOpticalFlowFarneback]
# (https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af). 
# Check out the description to see what all the different parameters does, and try to change them to see the difference.
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.5, 0)
# The function returns an array containing the flow vector for every pixel. 
# This can be changed to a magnitude and an angle using the function cv2.cartToPolar. 

# Find a way to represent the flow in the image. For example by drawing vectors the relevant places 
# or by making a new image with colors representing the flow of every pixel.
# An example of retrieving the magnitude and angle is shown below, which you can use if you like.
mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Retrieving the magnitude and angle of every pixel

### Fill in some code to represent the flow here ###
old_shape = img1.shape
prev_img_gray = gray1
current_img = img2
current_img_gray= gray2
assert current_img.shape == old_shape # assert: return TRUE if the conditions is true
hsv = np.zeros_like(img1)
hsv[...,1] = 255 # all the element of the first column set to 255
flow = None
flow = cv2.calcOpticalFlowFarneback(prev=prev_img_gray,
                                      next=current_img_gray, flow=flow,
                                      pyr_scale=0.8, levels=15, winsize=5,
                                      iterations=10, poly_n=5, poly_sigma=0,
                                      flags=10)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[..., 0] = ang*180/np.pi/2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
cv2.imshow(img2,rgb)
plt.show()