import sys
import os
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import cv2
import glob
import math
from functools import partial
from scipy.optimize import least_squares
from scipy.optimize import newton
import time

IMAGES_PATH = "underwater1100/"
IMAGES_TYPE = "*.png"

SQUARE_SIZE = 0.03 #meters
BOARD_HEIGHT = 6
BOARD_WIDTH = 8

INIT_CAMERA_CENTER_OFFSET = np.array([0., 0., -0.01])

def ray_trace_residual(undistorted_imgpoint, reprojected_imgpoint, objpoint, chessboard_normal, chessboard_origin, cal_params, dome_params, camera_center_offset):
    r_in, r_out, n_a, n_g, n_w = dome_params
    K, D = cal_params

    # air-glass intersection
    reprojected_imgpoint_h = np.concatenate((reprojected_imgpoint, [1]))

    ray_air = np.matmul(inv(K), reprojected_imgpoint_h.transpose())
    ray_air = ray_air / LA.norm(ray_air)
    # print("ray_air: ", ray_air)

    a = sum(ray_air * ray_air)
    b = 2 * sum(ray_air * camera_center_offset)
    c = sum(camera_center_offset * camera_center_offset) - r_in * r_in
    # print("a,b,c", a, b, c)

    k_air = np.roots([a, b, c])
    # print("k_air: ", k_air)
    k_air = k_air[0] if k_air[0] > 0 else k_air[1]
    # print("k_air: ", k_air)

    P_air_glass = camera_center_offset + k_air * ray_air
    # print("P_air_glass: ", P_air_glass)

    # refraction inside glass
    normal_air_glass = -P_air_glass
    theta_air = math.acos(sum(-ray_air * normal_air_glass))
    sin_theta_glass = n_a / n_g * math.sin(theta_air)
    if sin_theta_glass > 1:
        sin_theta_glass = 1
    ray_glass = n_a / n_g * ray_air + (n_a / n_g * math.cos(theta_air) - math.sqrt(1 - sin_theta_glass ** 2)) * normal_air_glass
    ray_glass = ray_glass / LA.norm(ray_glass)
    # print("ray_glass: ", ray_glass)

    # glass water intersection
    a = sum(ray_glass * ray_glass)
    b = 2 * sum(ray_glass * P_air_glass)
    c = sum(P_air_glass * P_air_glass) - r_out * r_out
    # print("a,b,c", a, b, c)

    k_glass = np.roots([a, b, c])
    # print("k_glass: ", k_glass)
    k_glass = k_glass[0] if k_glass[0] > 0 else k_glass[1]
    # print("k_glass: ", k_glass)

    P_glass_water = P_air_glass + k_glass * ray_glass
    # print("P_glass_water: ", P_glass_water)

    # refraction inside water
    normal_glass_water = -P_glass_water
    theta_glass_in = math.acos(sum(-ray_glass * normal_glass_water))
    # print("theta_glass_in", theta_glass_in)
    sin_theta_water = n_g / n_w * math.sin(theta_glass_in)
    if sin_theta_water > 1:
        sin_theta_water = 1
    # print("sin_theta_water: ", sin_theta_water)
    ray_water = n_g / n_w * ray_glass + (n_g / n_w * math.cos(theta_glass_in) - math.sqrt(1 - sin_theta_water ** 2)) * normal_glass_water
    ray_water = ray_water / LA.norm(ray_water)
    # print("ray_water: ", ray_water)

    # intersection with board
    t = sum(chessboard_normal * (chessboard_origin - P_glass_water)) / sum(chessboard_normal  * ray_water)
    P_chessboard = P_glass_water + ray_water * t
    # print("P_chessboard: ", P_chessboard, t)

    residual = P_chessboard - objpoint
    # print("objpoint: ", objpoint)
    # print("residual: ", residual)

    return LA.norm(residual)

def ray_trace_residuals(imgpoints, objpoints, rvec, tvec, cal_params, dome_params, camera_center_offset, reprojected_imgpoints):
    K, D = cal_params
    undistorted_imgpoints = cv2.undistortPoints(imgpoints, K, D)

    rvec = rvec.flatten()
    M_ex = np.zeros((4,4))
    R,J = cv2.Rodrigues(rvec)
    M_ex[:3,:3] = R
    M_ex[:3,3] = tvec.flatten()
    M_ex[3,:] = np.array([0, 0, 0, 1])
    # M_ex = inv(M_ex)
    # print("M: ", M_ex)

    # print(objpoints)
    chessboard_origin = M_ex[:3, 3] + camera_center_offset

    # compute chessboard normal
    z_axis_point_chessboard = np.array([0, 0, 1, 1])
    z_axis_point_cam = np.matmul(M_ex, z_axis_point_chessboard)
    z_axis_point_cam = z_axis_point_cam[:3]
    chessboard_normal = z_axis_point_cam - chessboard_origin
    # print("normal: ", chessboard_normal)

    total_residual = 0
    all_residual = []
    for cornerid in range(len(undistorted_imgpoints)):
    # for cornerid in range(0, len(undistorted_imgpoints), 10):
    # for cornerid in [3]:
        objpoint = np.concatenate((objpoints[cornerid], [1]))
        objpoint_cam = np.matmul(M_ex, objpoint)
        objpoint_cam = objpoint_cam[:3] + camera_center_offset

        reprojected_imgpoint = reprojected_imgpoints[2*(cornerid): 2*(cornerid+1)]
        # print(cornerid, reprojected_imgpoint)
        residual = ray_trace_residual(undistorted_imgpoints[cornerid], reprojected_imgpoint, objpoint_cam, chessboard_normal, chessboard_origin, cal_params, dome_params, camera_center_offset)
        total_residual += residual
        all_residual.append(residual)
        # print(cornerid, residual)
    mean_residual = total_residual / len(all_residual)

    # print("3d position total residual: ", total_residual)
    # print("3d position mean residual: ", mean_residual)
    return total_residual

class UW_DOME_CALIBRATION:
    def __init__(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((1, BOARD_HEIGHT*BOARD_WIDTH, 3), np.float32)
        objp[0,:,:2] = np.mgrid[0 : BOARD_WIDTH, 0 : BOARD_HEIGHT].T.reshape(-1,2) * SQUARE_SIZE

        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.img_shape = None

        # Parse chessboards from images
        images = glob.glob(IMAGES_PATH + IMAGES_TYPE)
        print("Parsing chessboards")

        for fname in images:
            img = cv2.imread(fname)
            self.img_shape = img.shape[:2]
            self.h, self.w = img.shape[:2]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (BOARD_WIDTH,BOARD_HEIGHT),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(objp)

                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners)

        print("Calibrating...")

        N_OK = len(self.objpoints)
        DIM = self.img_shape
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM = " + str(self.img_shape[::-1]))

        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))
        ret, self.K, self.D, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        print("K = np.array(" + str(self.K.tolist()) + ")")
        print("D = np.array(" + str(self.D.tolist()) + ")")

        self.n_w = 1.333
        self.n_a = 1.0
        self.n_g = 1.495
        self.r_in = 43.552 / 1000 # m
        self.r_out = 47.302 / 1000 # m

        self.imgpoints = np.squeeze(self.imgpoints)
        self.objpoints = np.squeeze(self.objpoints)

        # compute chessborad transformation
        # self.rvecs2 = []
        # self.tvecs2 = []
        # for img_id in range(len(self.imgpoints)):
        #     ret, rvec, tvec = cv2.solvePnP(self.objpoints[img_id], self.imgpoints[img_id], self.K, self.D, flags=cv2.SOLVEPNP_EPNP)
        #     self.rvecs2.append(rvec)
        #     self.tvecs2.append(tvec)

            # print("img_id: {}, rvec_diff:{}, tvec_diff:{}".format(img_id, self.rvecs[img_id] - rvec, self.tvecs[img_id] - tvec))

    def compute_air_calibration_residual(self):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.K, self.D)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error / len(self.objpoints)) )
        return mean_error

    def find_camera_center_offset(self, img_id = 3):
        init_camera_center_offset = INIT_CAMERA_CENTER_OFFSET
        print("initial camera center offset: ", init_camera_center_offset)
        cal_params = [self.K, self.D]
        dome_params = [self.r_in, self.r_out, self.n_a, self.n_g, self.n_w]

        low_bound = [-self.r_in/2, -self.r_in/2, -self.r_in]
        high_bound = [self.r_in/2, self.r_in/2, self.r_in]
        bound = (low_bound, high_bound)

        print("------------------------------------")
        print("Finding best camera center offset...")
        reprojection_imgpoints = partial(self.find_best_reprojection_imgpoints, self.imgpoints, self.objpoints, self.rvecs, self.tvecs, cal_params, dome_params)

        # residual = reprojection_imgpoints(init_camera_center_offset)
        # print(residual)

        # result = least_squares(reprojection_imgpoints, init_camera_center_offset, bounds=bound, ftol=1e-5, diff_step=1e-3)
        # result = newton(reprojection_imgpoints, init_camera_center_offset)
        result = least_squares(reprojection_imgpoints, init_camera_center_offset, ftol=1e-5, method='lm', diff_step=1e-4)
        print("optimized camera center offset: ", result.x)

    def find_best_reprojection_imgpoints(self, imgpoints, objpoints, rvecs, tvecs, cal_params, dome_params, camera_center_offset):
        total_residual = 0
        all_residual = []
        print("------------------------------------")
        print("camera center offset: ", camera_center_offset)

        # for img_id in range(0, len(imgpoints), 10):
        for img_id in [2]:
            # print("img_id: ", img_id)

            minimize_residuals = partial(ray_trace_residuals, imgpoints[img_id], objpoints[img_id], rvecs[img_id], tvecs[img_id], cal_params, dome_params, camera_center_offset)

            init_reprojected_imgpoints = []
            for cornerpoints in imgpoints[img_id]:
                init_reprojected_imgpoints.extend(cornerpoints)

            # minimize_residuals(init_reprojected_imgpoints)

            print("Finding best reprojected position...")
            init_time = time.time()
            result = least_squares(minimize_residuals, init_reprojected_imgpoints, ftol=1e-6)
            best_reproj_imgpoints = result.x
            # print(best_reproj_imgpoints)
            # print("3d position mean residual: ", result.cost)

            # compute image points residual
            for i, cornerpoints in enumerate(imgpoints[img_id]):
                reproject_imgpoints = best_reproj_imgpoints[2 * i: 2 * (i + 1)]
                residual = reproject_imgpoints - cornerpoints
                all_residual.extend(residual)
                total_residual += LA.norm(residual)
                # print("imgpoint: ", cornerpoints)
                # print("reprojected imgpoint: ", reproject_imgpoints)
                # print("cornerid: {}, residual: {}, residual_norm: {}".format(i, residual, LA.norm(residual)))

        mean_residual = total_residual / len(all_residual)
        print("mean reprojection residual: ", mean_residual)
        print("total reprojection residual: ", total_residual)
        print("time to find best reprojected position: ", time.time() - init_time, "s")
        print("------------------------------------")
        # print(all_residual)

        return all_residual
        # return total_residual


if __name__ == "__main__":
    init_time = time.time()
    uw = UW_DOME_CALIBRATION()
    uw.find_camera_center_offset()

    print("time: ", time.time() - init_time, "s")
    