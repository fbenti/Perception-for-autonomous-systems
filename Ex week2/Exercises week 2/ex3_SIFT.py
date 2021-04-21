import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Book.jpg")
# b,g,r = cv2.split(img) # Changing the order from bgr to rgb so that matplotlib can show it
# img = cv2.merge([r,g,b])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

# We'll start by converting the image to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray, cmap = 'gray')
# plt.show()

###
# Now we will attempt to find the features and descriptors using [SIFT](https://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html). 
# In OpenCV we will do it by first creating a SIFT object and then use the detectAndCompute function.
sift = cv2.xfeatures2d.SIFT_create() # create a SIFT object
kp, des = sift.detectAndCompute(gray, None) # find Keypoints and descriptors

# To show the detected keypoints we use the function [drawKeyPoints]
# (https://docs.opencv.org/master/d4/d5d/group__features2d__draw.html#ga5d2bafe8c1c45289bc3403a40fb88920).
#  We scale the picture up a bit using "figsize", so it's easier to see.
kp_img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # flags= For each keypoint the circle around keypoint with keypoint size and orientation will be drawn.
# plt.figure(figsize = (10,10))
# plt.imshow(kp_img)
# plt.show()

# Let's load the image "More_books.jpg" and convert it to grayscale.
img2 = cv2.imread("More_books.jpg")
b,g,r = cv2.split(img2) # Changing the order from bgr to rgb so that matplotlib can show it
img2 = cv2.merge([r,g,b])
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray2, cmap = 'gray')

# Now we use the same SIFT method to find the keypoints and descriptors of this image.
kp2, des2 = sift.detectAndCompute(gray2, None)
kp_img2 = cv2.drawKeypoints(gray2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# plt.imshow(kp_img2)

# Now that we have the descriptors of both images, let's see if we can match them and find the book from the first image in the second image.
# For this example we use Brute Force Matcher 
# ([cv2.BFMatcher](https://docs.opencv.org/master/d3/da1/classcv_1_1BFMatcher.html)).
#  We use the function cv2.match to find the matches.
bf = cv2.BFMatcher()
matches = bf.match(des, des2)

# We use the function [cv2.drawMatches](https://docs.opencv.org/3.4/d4/d5d/group__features2d__draw.html) to display the result. 
# We'd like to display the best of the matches. The matching is made using a distance measurement between the descriptors,
# the smaller the distance, the better the match. So we sort the "matches" array using the distance term and then we plot only the first 100 matches.
matches = sorted(matches, key = lambda x:x.distance)
# 4th arg: Matches from the first image to the second one, which means that keypoints1[i] has a corresponding point in keypoints2[matches[i]] .
img3 = cv2.drawMatches(gray,kp,gray2,kp2,matches[:700],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize = (15,15))
plt.title('SIFT')
plt.imshow(img3)
plt.show()