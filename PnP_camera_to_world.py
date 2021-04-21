import cv2 as cv2
import numpy as np


rvec = np.array((-0.05, -1.51, -0.00)).reshape(3,1)
tvec = np.array((87.39, -2.25, -24.89)).reshape(3,1)
XYZ_camera_coor = np.array((-6.71, 0.023, 21.59, 1)).reshape(4,1)
print(rvec.shape)
print(tvec.shape)


# FROM CAMERA coordinates to 3D world coordinates
rot, _ = cv2.Rodrigues(rvec)
print("rot: ", rot.shape)
tvec = -rot.T.dot(tvec)  # coordinate transformation, from camera to world. What is the XYZ of the camera wrt World
print("tvec: ", tvec.shape)
inv_transform = np.hstack((rot.T, tvec))  # inverse transform. A transform projecting points from the camera frame to the world frame
print("inv_transform: ", inv_transform.shape)


XYZ_world_coor = np.dot(inv_transform, XYZ_camera_coor)
print("inv_transform: ", inv_transform.shape)
print(XYZ_world_coor)    