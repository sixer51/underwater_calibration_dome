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

# set underwater distorted image points and object depth
# use ray model found 3d object points
# project object points to get in air image points

CAMERA_CENTER_OFFSET = np.array([0.0, 0.0, 0.005])

HEIGHT = 1080
WIDTH = 1440
IMG_INTERVAL = 40
NUM_X = WIDTH // IMG_INTERVAL + 1
NUM_Y = HEIGHT // IMG_INTERVAL + 1
DEPTH = 1.5

K = np.array([[2500, 0.0, 720], [0.0, 2500, 540], [0.0, 0.0, 1.0]])
# D = np.array([[0., 0., 0., 0., 0.0]])
# K = np.array([[2497.545748328383, 0.0, 775.9114854173114], [0.0, 2505.3910190565584, 531.934918611321], [0.0, 0.0, 1.0]])
D = np.array([[-0.37595631549789615, 0.026558128859013045, 0.004159110760200319, -0.0015148256923774017, 0.0]])

n_w = 1.333
n_a = 1.0
n_g = 1.495
r_in = 43.552 / 1000 # m
r_out = 47.302 / 1000 # m

cal_params = [K, D]
dome_params = [r_in, r_out, n_a, n_g, n_w]

def ray_trace_residual(distorted_imgpoint, cal_params, dome_params, camera_center_offset, depth):
    r_in, r_out, n_a, n_g, n_w = dome_params
    K, D = cal_params

    # air-glass intersection
    distorted_imgpoint_h = np.concatenate((distorted_imgpoint, [1]))

    ray_air = np.matmul(inv(K), distorted_imgpoint_h.transpose())
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

    # find intersection with depth plane
    t = (DEPTH + camera_center_offset[2] - P_glass_water[2]) / ray_water[2]
    objpoint = P_glass_water + t * ray_water

    return objpoint - camera_center_offset

def ray_trace_residuals(distorted_imgpoints, cal_params, dome_params, camera_center_offset, depth = DEPTH):
    # compute_reprojection_residual(imgpoints, reprojected_imgpoints)

    # K, D = cal_params
    # distorted_imgpoints = cv2.undistortPoints(distorted_imgpoints, K, D)
    # distorted_imgpoints = np.squeeze(distorted_imgpoints)

    distorted_imgpoints = np.squeeze(distorted_imgpoints)
    objpoints = []
    for cornerid in range(len(distorted_imgpoints)):
        objpoint = ray_trace_residual(distorted_imgpoints[cornerid], cal_params, dome_params, camera_center_offset, depth)
        objpoints.append(objpoint)
        # print(cornerid / (NUM_X * NUM_Y) * 100, "%")

    return np.array(objpoints, dtype=np.float32)

def compute_corner_position(imgpoint, chessboard_normal, chessboard_origin, cal_params, dome_params, camera_center_offset):
    r_in, r_out, n_a, n_g, n_w = dome_params
    K, D = cal_params

    # air-glass intersection
    imgpoint_h = np.concatenate((imgpoint, [1]))

    ray_air = np.matmul(inv(K), imgpoint_h.transpose())
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
    t = np.dot(chessboard_normal, chessboard_origin - P_glass_water) / np.dot(chessboard_normal, ray_water)
    P_chessboard = P_glass_water + ray_water * t
    # print("P_chessboard: ", P_chessboard, t)

    return P_chessboard

def compute_image_corners_positions(imgpoints, rvec, tvec, cal_params, dome_params, camera_center_offset):
    # K, D = cal_params
    # imgpoints = cv2.undistortPoints(imgpoints, K, D)

    rvec = rvec.flatten()
    M_ex = np.zeros((4,4))
    R,J = cv2.Rodrigues(rvec)
    M_ex[:3,:3] = R
    M_ex[:3,3] = tvec.flatten()
    M_ex[3,:] = np.array([0, 0, 0, 1])
    chessboard_origin = M_ex[:3, 3] + camera_center_offset

    # compute chessboard normal
    z_axis_point_chessboard = np.array([0, 0, 1, 1])
    z_axis_point_cam = np.matmul(M_ex, z_axis_point_chessboard)
    z_axis_point_cam = z_axis_point_cam[:3]
    chessboard_normal = z_axis_point_cam - chessboard_origin
    chessboard_normal = chessboard_normal / LA.norm(chessboard_normal)
    # print("normal: ", chessboard_normal)

    obj_corner_points = []
    for cornerid in range(len(imgpoints)):
        objpoint = compute_corner_position(imgpoints[cornerid], chessboard_normal, chessboard_origin, cal_params, dome_params, camera_center_offset)
        obj_corner_points.append(objpoint)
    obj_corner_points = np.array([obj_corner_points], dtype=np.float32)

    return obj_corner_points

def generate_points_for_calibration(camera_center_offset):
    NUM_IMAGE = 100
    NUM_POINT = 60

    imgpoints = []
    objpoints = []
    for _ in range(NUM_IMAGE):
        img_corners = np.random.rand(NUM_POINT, 2)
        img_corners[:, 0] *= WIDTH
        img_corners[:, 1] *= HEIGHT
        img_corners = np.array(img_corners, dtype=np.float32)
        imgpoints.append(img_corners)

        # plt.scatter(img_corners[:, 0], img_corners[:, 1])
        # plt.show()

        rvec = np.random.rand(3)
        rvec -= 0.5
        tvec = np.random.rand(3)
        tvec[:2] -= 0.5
        tvec[2] += 0.5

        # get obj_points
        obj_corner_points = compute_image_corners_positions(img_corners, rvec, tvec, cal_params, dome_params, camera_center_offset)
        objpoints.append(obj_corner_points)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # obj_corner_points = np.squeeze(obj_corner_points)
        # ax.scatter(obj_corner_points[:, 0], obj_corner_points[:, 1], obj_corner_points[:, 2])
        # plt.show()

    imgpoints = np.array(imgpoints)
    objpoints = np.array(objpoints)
    # print(imgpoints)
    # print(objpoints)
    return imgpoints, objpoints

def calibrate_underwater_pinhole(camera_center_offset):
    imgpoints, objpoints = generate_points_for_calibration(camera_center_offset)
    ret, K_calib, D_calib, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1080, 1440), K, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    print("K_underwater = ", K_calib)
    print("D_underwater = ", D_calib)
    # print("ret = ", ret)
    # print("tvec = ", tvecs)
    return K_calib, D_calib

def get_index(x, y):
    # return y * NUM_X + x
    return y * NUM_X + x

def get_coordinate(index):
    y = index // NUM_X
    x = index % NUM_X
    return x, y

def compute_corresponding_imgpoints(camera_center_offset):
    print("camera center offset: ", camera_center_offset)

    # generate underwater distorted img_points
    distorted_imgpoints = []
    x_start = WIDTH % IMG_INTERVAL // 2
    y_start = HEIGHT % IMG_INTERVAL // 2

    for y in range(NUM_Y):
        for x in range(NUM_X):
            u = x_start + x * IMG_INTERVAL
            v = y_start + y * IMG_INTERVAL
            distorted_imgpoints.append(np.array([u, v], dtype=np.float32))
    distorted_imgpoints = np.array(distorted_imgpoints)
    # print(distorted_imgpoints)

    # find objpoints
    objpoints = ray_trace_residuals(distorted_imgpoints, cal_params, dome_params, camera_center_offset)

    # get obj_points
    # rvec = np.zeros(3)
    # tvec = np.array([0., 0., DEPTH])
    # objpoints = compute_image_corners_positions(distorted_imgpoints, rvec, tvec, cal_params, dome_params, camera_center_offset)

    # project objpoints to image
    imgpoints, _ = cv2.projectPoints(objpoints, np.zeros(3), np.zeros(3), K, D)
    imgpoints = cv2.undistortPoints(imgpoints, K, D, None, K)
    imgpoints = np.squeeze(imgpoints)

    # calibrate camera with underwater image
    # K_water, D_water = calibrate_underwater_pinhole(camera_center_offset)
    # imgpoints_water, _ = cv2.projectPoints(objpoints, np.zeros(3), np.zeros(3), K_water, D_water)
    # imgpoints_water = cv2.undistortPoints(imgpoints_water, K_water, D_water, None, K_water)
    # imgpoints_water = np.squeeze(imgpoints_water)
    # print(imgpoints)

    return distorted_imgpoints, imgpoints

def map_air_to_underwater(distorted_imgpoints, imgpoints, x, y):
    u_left = min(int(x // IMG_INTERVAL), NUM_X - 1)
    v_top = min(int(y // IMG_INTERVAL), NUM_Y - 1)

    # find u_left and u_right
    while x - imgpoints[get_index(u_left, v_top)][0] < 0:
        u_left -= 1
        if u_left == 0:
            break
    u_right = u_left if u_left == NUM_X - 1 else u_left + 1

    # find v_top and v_bottom
    while y - imgpoints[get_index(u_left, v_top)][1] < 0:
        v_top -= 1
        if v_top == 0:
            break
    v_bottom = v_top if v_top == NUM_Y - 1 else v_top + 1

    if u_left == u_right:
        mapped_x = distorted_imgpoints[get_index(u_left, v_top)][0]
    else:
        imgpoints_xl = imgpoints[get_index(u_left, v_top)][0]
        imgpoints_xr = imgpoints[get_index(u_right, v_top)][0]
        t_x = (x - imgpoints_xl) / (imgpoints_xr - imgpoints_xl)

        distort_xl = distorted_imgpoints[get_index(u_left, v_top)][0]
        distort_xr = distorted_imgpoints[get_index(u_right, v_top)][0]
        mapped_x = t_x * (distort_xr - distort_xl) + distort_xl

    if v_top == v_bottom:
        mapped_y = distorted_imgpoints[get_index(u_left, v_top)][1]
    else:
        imgpoints_yt = imgpoints[get_index(u_left, v_top)][1]
        imgpoints_yb = imgpoints[get_index(u_left, v_bottom)][1]
        t_y = (y - imgpoints_yt) / (imgpoints_yb - imgpoints_yt)

        distort_yt = distorted_imgpoints[get_index(u_left, v_top)][1]
        distort_yb = distorted_imgpoints[get_index(u_left, v_bottom)][1]
        mapped_y = t_y * (distort_yb - distort_yt) + distort_yt

    return mapped_x, mapped_y

def map_underwater_to_air(distorted_imgpoints, imgpoints, x, y):
    u_left = int(x // IMG_INTERVAL)
    v_top = int(y // IMG_INTERVAL)
    u_right = min(u_left + 1, NUM_X - 1)
    v_bottom = min(v_top + 1, NUM_Y - 1)

    left = u_left * IMG_INTERVAL
    right = u_right * IMG_INTERVAL
    top = v_top * IMG_INTERVAL
    bottom = v_bottom * IMG_INTERVAL
    # print(left, right, top, bottom)
    # print(imgpoints[get_index(u_left, v_top)], imgpoints[get_index(u_right, v_top)])

    if (left == right or x == left) and (top == bottom or y == top):
        mapped_cooridnate = imgpoints[get_index(u_left, v_top)]
    elif (left == right or x == left) and top != bottom:
        mapped_cooridnate = (bottom - y) * imgpoints[get_index(u_left, v_top)] + \
                        (y - top) * imgpoints[get_index(u_left, v_bottom)]
        mapped_cooridnate = np.array(mapped_cooridnate) / IMG_INTERVAL
    elif left != right and (top == bottom or y == top):
        mapped_cooridnate = (right - x) * imgpoints[get_index(u_left, v_top)] + \
                        (x - left) * imgpoints[get_index(u_right, v_top)]
        mapped_cooridnate = np.array(mapped_cooridnate) / IMG_INTERVAL
    else:
        mapped_cooridnate = (right - x) * (bottom - y) * imgpoints[get_index(u_left, v_top)] + \
                            (x - left) * (bottom - y) * imgpoints[get_index(u_right, v_top)] + \
                            (right - x) * (y - top) * imgpoints[get_index(u_left, v_bottom)] + \
                            (x - left) * (y - top) * imgpoints[get_index(u_right, v_bottom)]
        mapped_cooridnate = np.array(mapped_cooridnate) / (IMG_INTERVAL ** 2)
    

    return mapped_cooridnate

def generate_rectification_map(camera_center_offset):
    distorted_imgpoints, imgpoints = compute_corresponding_imgpoints(camera_center_offset)
    verifictaion(camera_center_offset, distorted_imgpoints, imgpoints)

    # plot objpoints
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # objpoints = np.squeeze(objpoints)
    # ax.scatter(objpoints[:, 0], objpoints[:, 1], objpoints[:, 2])

    # plot pixel difference
    # plt.figure()
    # interval = NUM_X // 10
    # plt.scatter(distorted_imgpoints[0:-1:interval, 0], distorted_imgpoints[0:-1:interval, 1])
    # plt.scatter(imgpoints[0:-1:interval, 0], imgpoints[0:-1:interval, 1], marker='*')
    # plt.scatter(imgpoints_water[0:-1:interval, 0], imgpoints_water[0:-1:interval, 1], marker='+')

    # plot residual map without calibration
    # plot_residuals(imgpoints, distorted_imgpoints)
    # plot_residuals(distorted_imgpoints, imgpoints, "inverse")

    # residual map after calibration
    # plot_residuals(imgpoints_water, distorted_imgpoints, "pinhole calibration")


    plt.show()

def plot_residuals(imgpoints, reference_imgpoints, title_prefix="", num_x=NUM_X, num_y=NUM_Y):
    residuals = []
    residuals_x = []
    residuals_y = []
    for i, point in enumerate(reference_imgpoints):
        dx = point[0] - imgpoints[i, 0]
        dy = point[1] - imgpoints[i, 1]
        residual = math.sqrt(dx**2 + dy**2)
        residuals.append(residual)
        residuals_x.append(dx)
        residuals_y.append(dy)

    residuals = np.array(residuals)
    residuals = residuals.reshape(num_y, num_x)
    residuals_x = np.array(residuals_x)
    residuals_x = residuals_x.reshape(num_y, num_x)
    residuals_y = np.array(residuals_y)
    residuals_y = residuals_y.reshape(num_y, num_x)

    fig, ax = plt.subplots()
    ax.set_title(title_prefix+" residual")
    im = ax.imshow(residuals)
    fig.colorbar(im)

    # fig, ax = plt.subplots()
    # ax.set_title("x")
    # im = ax.imshow(residuals_x)
    # fig.colorbar(im)

    # fig, ax = plt.subplots()
    # ax.set_title("y")
    # im = ax.imshow(residuals_y)
    # fig.colorbar(im)

    return residuals

def verifictaion(camera_center_offset, distorted_imgpoints, imgpoints):
    # distorted_imgpoints = np.random.rand(NUM_POINT, 2)
    # distorted_imgpoints[:, 0] *= WIDTH
    # distorted_imgpoints[:, 1] *= HEIGHT
    # distorted_imgpoints = np.array(distorted_imgpoints, dtype=np.float32)
    # generate underwater distorted img_points
    v_distorted_imgpoints = []
    interval = 10
    x_start = WIDTH % interval // 2
    y_start = HEIGHT % interval // 2
    num_x = WIDTH // interval + 1
    num_y = HEIGHT // interval + 1
    depth = 2.5

    for y in range(num_y):
        for x in range(num_x):
            u = x_start + x * interval
            v = y_start + y * interval
            v_distorted_imgpoints.append(np.array([u, v], dtype=np.float32))
    v_distorted_imgpoints = np.array(v_distorted_imgpoints)
    # print(v_distorted_imgpoints)

    # rvec = np.random.rand(3)
    # rvec -= 0.5
    # tvec = np.random.rand(3)
    # tvec[:2] -= 0.5
    # tvec[2] += 0.5
    rvec = np.zeros(3)
    tvec = np.array([0., 0., depth])

    # get obj_points
    # v_objpoints = compute_image_corners_positions(v_distorted_imgpoints, rvec, tvec, cal_params, dome_params, camera_center_offset)
    v_objpoints = ray_trace_residuals(v_distorted_imgpoints, cal_params, dome_params, camera_center_offset, depth)
    # v_objpoints_test = ray_trace_residuals(v_distorted_imgpoints, cal_params, dome_params, camera_center_offset)

    v_imgpoints, _ = cv2.projectPoints(v_objpoints, np.zeros(3), np.zeros(3), K, D)
    v_imgpoints = cv2.undistortPoints(v_imgpoints, K, D, None, K)
    v_imgpoints = np.squeeze(v_imgpoints)

    # remap imgpoints to distorted points
    remapped_distorted_imgpoints = []
    x_start = WIDTH % IMG_INTERVAL // 2
    y_start = HEIGHT % IMG_INTERVAL // 2

    # for i, point in enumerate(distorted_imgpoints):
    for i, point in enumerate(v_distorted_imgpoints):
        u = point[0]
        v = point[1]
        mapped_cooridnate = map_underwater_to_air(distorted_imgpoints, imgpoints, u, v)
        # u_remapped, v_remapped = map_air_to_underwater(distorted_imgpoints, imgpoints, u, v)
        # print("u, v: ", u, v)
        # print(mapped_cooridnate)
        # print(v_imgpoints[i])
        remapped_distorted_imgpoints.append(mapped_cooridnate)
    remapped_distorted_imgpoints = np.array(remapped_distorted_imgpoints)
    # print(remapped_distorted_imgpoints)

    # calibrate camera with underwater image
    K_water, D_water = calibrate_underwater_pinhole(camera_center_offset)
    imgpoints_water, _ = cv2.projectPoints(v_objpoints, np.zeros(3), np.zeros(3), K_water, D_water)
    imgpoints_water = cv2.undistortPoints(imgpoints_water, K_water, D_water, None, K_water)
    imgpoints_water = np.squeeze(imgpoints_water)

    # plot_residuals(remapped_distorted_imgpoints, imgpoints, title_prefix="remapped")
    plot_residuals(remapped_distorted_imgpoints, v_imgpoints, num_x=num_x, num_y=num_y, title_prefix="refraction")
    plot_residuals(v_distorted_imgpoints, v_imgpoints, num_x=num_x, num_y=num_y, title_prefix="no calibration")
    plot_residuals(imgpoints_water, v_distorted_imgpoints, num_x=num_x, num_y=num_y, title_prefix="pinhole")
    # plot_residuals(v_distorted_imgpoints, distorted_imgpoints, num_x=num_x, num_y=num_y, title_prefix="test")
    # plot_residuals(remapped_distorted_imgpoints, v_distorted_imgpoints, num_x=num_x, num_y=num_y)


if __name__ == "__main__":
    init_time = time.time()

    # camera_center_offset = CAMERA_CENTER_OFFSET
    # interval = 0.0025
    # for i in range(5):
    #     camera_center_offset[2] = float(i * interval)
    #     generate_rectification_map(camera_center_offset)
    generate_rectification_map(CAMERA_CENTER_OFFSET)

    print("time: ", time.time() - init_time, "s")