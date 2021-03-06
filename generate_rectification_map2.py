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
from matplotlib import pyplot as plt

# set object points in camera frame
# project object points to image plane to get image points
# 3d position error = distance between objpoints and ray_water

INIT_CAMERA_CENTER_OFFSET = np.array([0.0, 0., 0.0])

HEIGHT = 1080
WIDTH = 1440
NUM_X = 16
NUM_Y = int(NUM_X * 3 / 4)

def ray_trace_residual(undistorted_imgpoint, reprojected_imgpoint, objpoint, cal_params, dome_params, camera_center_offset):
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
    # ray_water = ray_water / LA.norm(ray_water)
    # print("ray_water: ", ray_water)

    # compute distance between objpoint and ray
    # print(objpoint - P_glass_water)
    projected_distance = np.dot(objpoint - P_glass_water, ray_water) / LA.norm(ray_water)
    hypotenuse_len_square = np.dot(objpoint - P_glass_water, objpoint - P_glass_water)
    distance_diff = hypotenuse_len_square - projected_distance ** 2
    # residual = 0 if distance_diff < 0 else math.sqrt(distance_diff)
    residual = math.sqrt(abs(distance_diff))

    return residual

def compute_reprojection_residual(imgpoints, reprojected_imgpoints):
    total_residual = 0
    for i, cornerpoints in enumerate(imgpoints):
        reproject_imgpoints = reprojected_imgpoints[2 * i: 2 * (i + 1)]
        residual = reproject_imgpoints - cornerpoints
        total_residual += LA.norm(residual)
    mean_residual = total_residual / len(imgpoints)
    print("mean reprojected residual: ", mean_residual)

def ray_trace_residuals(imgpoints, objpoints, cal_params, dome_params, camera_center_offset, reprojected_imgpoints):
    compute_reprojection_residual(imgpoints, reprojected_imgpoints)

    K, D = cal_params
    undistorted_imgpoints = cv2.undistortPoints(imgpoints, K, D)

    total_residual = 0
    all_residual = []
    for cornerid in range(len(undistorted_imgpoints)):
        reprojected_imgpoint = reprojected_imgpoints[2*(cornerid): 2*(cornerid+1)]
        # print(cornerid, reprojected_imgpoint)
        residual = ray_trace_residual(undistorted_imgpoints[cornerid], reprojected_imgpoint, objpoints[cornerid] + camera_center_offset, cal_params, dome_params, camera_center_offset)
        total_residual += residual
        all_residual.append(residual)
        # print(cornerid, residual)
    mean_residual = total_residual / len(all_residual)

    print("3d position total residual: ", total_residual)
    # print("3d position mean residual: ", mean_residual)
    return total_residual
    

def find_best_reprojection_imgpoints(imgpoints, objpoints, cal_params, dome_params, camera_center_offset):
    total_residual = 0
    all_residual = []
    print("------------------------------------")
    print("camera center offset: ", camera_center_offset)

    minimize_residuals = partial(ray_trace_residuals, imgpoints, objpoints, cal_params, dome_params, camera_center_offset)

    init_reprojected_imgpoints = []
    for cornerpoints in imgpoints:
        init_reprojected_imgpoints.extend(cornerpoints)

    minimize_residuals(init_reprojected_imgpoints)

    print("Finding best reprojected position...")
    init_time = time.time()
    result = least_squares(minimize_residuals, init_reprojected_imgpoints, ftol=0.01, diff_step=1e-6)
    best_reproj_imgpoints = result.x
    # print(best_reproj_imgpoints)
    # print("3d position mean residual: ", result.cost)

    # compute image points residual
    reprojected_imgpoints = []
    for i, cornerpoints in enumerate(imgpoints):
        reproject_imgpoints = best_reproj_imgpoints[2 * i: 2 * (i + 1)]
        residual = reproject_imgpoints - cornerpoints
        # all_residual.extend(residual)
        all_residual.append(LA.norm(residual))
        total_residual += LA.norm(residual)
        # print("imgpoint: ", cornerpoints)
        # print("reprojected imgpoint: ", reproject_imgpoints)
        # print("cornerid: {}, residual: {}, residual_norm: {}".format(i, residual, LA.norm(residual)))
        reprojected_imgpoints.append(reproject_imgpoints)

    mean_residual = total_residual / len(all_residual)
    print("mean reprojection residual: ", mean_residual)
    print("total reprojection residual: ", total_residual)
    print("time to find best reprojected position: ", time.time() - init_time, "s")
    print("------------------------------------")
    # print(all_residual)

    return all_residual, reprojected_imgpoints
    # return total_residual

def generate_rectification_map(img_id = 3):
    init_camera_center_offset = INIT_CAMERA_CENTER_OFFSET
    print("camera center offset: ", init_camera_center_offset)

    K = np.array([[2497.545748328383, 0.0, 775.9114854173114], [0.0, 2505.3910190565584, 531.934918611321], [0.0, 0.0, 1.0]])
    D = np.array([[-0.37595631549789615, 0.026558128859013045, 0.004159110760200319, -0.0015148256923774017, 0.0]])
    # K = np.array([[2497.545748328383, 0.0, WIDTH / 2], [0.0, 2505.3910190565584, HEIGHT / 2], [0.0, 0.0, 1.0]])
    # K = np.array([[2500.0, 0.0, WIDTH / 2], [0.0, 2500.0, HEIGHT / 2], [0.0, 0.0, 1.0]])
    # D = np.array([[0., 0., 0., 0., 0.0]])

    n_w = 1.333
    n_a = 1.0
    n_g = 1.495
    r_in = 43.552 / 1000 # m
    r_out = 47.302 / 1000 # m

    cal_params = [K, D]
    dome_params = [r_in, r_out, n_a, n_g, n_w]

    # generate imgpoints and objpoints
    imgpoints = []
    objpoints = []
    obj_interval = 0.03 # m
    num_x = NUM_X
    num_y = NUM_Y
    depth = 1.2
    x_start = -obj_interval * (num_x / 2 - 0.5)
    y_start = -obj_interval * (num_y / 2 - 0.5)

    for y in range(num_y):
        for x in range(num_x):
            objpoint = np.array([x_start + x * obj_interval, y_start + y * obj_interval, depth], dtype=np.float32)
            objpoints.append(objpoint)

    objpoints = np.array(objpoints)
    imgpoints, _ = cv2.projectPoints(objpoints, np.zeros(3), np.zeros(3), K, D)
    imgpoints = np.squeeze(imgpoints)
    # print(imgpoints)
    # print(objpoints)

    reprojection_imgpoints = partial(find_best_reprojection_imgpoints, imgpoints, objpoints, cal_params, dome_params)

    # reprojection_imgpoints(init_camera_center_offset)
    residual, reprojected_imgpoints = reprojection_imgpoints(init_camera_center_offset)
    # print(residual)

    plt.figure()
    imgpoints = np.squeeze(imgpoints)
    reprojected_imgpoints = np.array(reprojected_imgpoints)
    plt.scatter(imgpoints[:, 0], imgpoints[:, 1])
    plt.scatter(reprojected_imgpoints[:, 0], reprojected_imgpoints[:, 1], marker='*')

    return residual, reprojected_imgpoints



if __name__ == "__main__":
    init_time = time.time()
    residual, reprojected_imgpoints = generate_rectification_map()
    # generate_rectification_map()

    print("time: ", time.time() - init_time, "s")
    
    residual = np.array(residual)
    # print(residual)
    residual = residual.reshape(NUM_Y, NUM_X)

    fig, ax = plt.subplots()
    im = ax.imshow(residual)
    fig.colorbar(im)

    plt.show()