import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Book.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Create SURF object
surf = cv2.xfeatures2d.SURF_create()
kp, des = surf.detectAndCompute(img,None)

kp_img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize = (10,10))
# plt.imshow(kp_img)
# plt.show()

img2 = cv2.imread("More_books.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

kp2, des2 = surf.detectAndCompute(img2,None)
kp_img2 = cv2.drawKeypoints(gray2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# plt.imshow(kp_img2)
# plt.show()

bf = cv2.BFMatcher()
matches = bf.match(des, des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(gray,kp,gray2,kp2,matches[:700],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.title('SURF')
plt.imshow(img3)
plt.show()