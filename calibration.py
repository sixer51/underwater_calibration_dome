import sys
import os
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import math

# IMAGES_PATH = "test_images/"
IMAGES_PATH = "underwater1100/"
IMAGES_TYPE = "*.png"
UNDISTORT_OUTFOLDER = "output/"

SQUARE_SIZE = 0.03 #meters
BOARD_HEIGHT = 6
BOARD_WIDTH = 8

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1,BOARD_HEIGHT*BOARD_WIDTH,3), np.float32)
objp[0,:,:2]  = np.mgrid[0:BOARD_WIDTH,0:BOARD_HEIGHT].T.reshape(-1,2) * SQUARE_SIZE

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
img_shape = None

# Parse chessboards from images
images = glob.glob(IMAGES_PATH + IMAGES_TYPE)

print("Parsing chessboards")

for fname in images:
    img = cv2.imread(fname)
    img_shape = img.shape[:2]
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_WIDTH,BOARD_HEIGHT),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (BOARD_WIDTH,BOARD_HEIGHT), corners,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(500)

# cv2.destroyAllWindows()

print("Calibrating...")

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros(4)

DIM = img_shape
print("Found " + str(N_OK) + " valid images for calibration")

# ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("DIM = " + str(img_shape[::-1]))
print("K = np.array(" + str(K.tolist()) + ")")
print("D = np.array(" + str(D.tolist()) + ")")
# D[0, 4] = 0.0
# print("rotation vectors: " + str(rvecs))
# print(len(rvecs))
# print("translation vectors: " + str(tvecs))
# print(len(tvecs))

print(imgpoints[0])
print(objpoints)
# print(rvecs[0])
# print(tvecs[0])

# Check the reprojection errors
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error / len(objpoints)) )

# Show the undistorted images
# print("Saving undistorted images...")
# try:
#     os.mkdir(UNDISTORT_OUTFOLDER)
# except OSError as oe:
#     print("Warning: output folder already exists, may be overwriting images.")

# K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
# print(mapx, mapy)
# print(len(mapx[0]))

# fig,ax=plt.subplots(figsize=(10,20))
# im=ax.imshow(score,cmap='plasma_r') #用cmap设置配色方案
# ax.xaxis.set_ticks_position('top') #设置x轴刻度到上方
# ax.set_xticks(np.arange(len(col))) #设置x轴刻度
# ax.set_yticks(np.arange(len(name))) #设置y轴刻度
# ax.set_xticklabels(mapx) #设置x轴刻度标签
# ax.set_yticklabels(mapy) #设置y轴刻度标签
# fig.colorbar(im,pad=0.03) #设置颜色条

residual = np.zeros((h, w))
for v in range(h):
    for u in range(w):
        # if u in range(0, w, 200) and v in range(0, h, 200):
            # print(mapx[v, u], u)
            # print(mapy[v, u], v)
        # print(mapx[v, u] - u, mapy[v, u] - v)
        residual[v, u] = math.sqrt((mapx[v, u] - u)**2 + (mapy[v, u] - v)**2)

fig, ax = plt.subplots()
im = ax.imshow(residual)
# ax.set_xticks(np.arange(w))
# ax.set_yticks(np.arange(h))
fig.colorbar(im)

plt.show()


# for fname in images:
#     img = cv2.imread(fname)
#     h, w = img.shape[:2]

#     prefix = fname[len(IMAGES_PATH):-4]

#     # undistort
#     dst = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

    # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv2.imwrite(UNDISTORT_OUTFOLDER + prefix + '_undistorted.png', dst)

#     cv2.imshow('img', dst)
#     cv2.waitKey(100)

# cv2.destroyAllWindows()