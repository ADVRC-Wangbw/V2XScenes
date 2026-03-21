import cv2
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import pickle

def filter_point_cloud_by_xyz(pcd_path, x_min, x_max, y_min, y_max, z_min, z_max):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    mask = (
    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
    (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    filtered_points = points[mask]

    return filtered_points

def transform_matrix(x, y, z, qx, qy, qz, qw):
    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [x, y, z]
    return transformation_matrix

def read_pcd(pcd_file_path):
    points = []
    intensities = []

    with open(pcd_file_path, 'r') as f:
        lines = f.readlines()

    # 寻找数据开始的行
    data_start_line = 0
    for i, line in enumerate(lines):
        if line.startswith("DATA"):
            data_start_line = i + 1
            break

    # 从数据开始的行解析点云数据
    for line in lines[data_start_line:]:
        data = line.strip().split()
        if len(data) >= 3: # 至少包含x、y、z坐标
            points.append([float(data[0]), float(data[1]), float(data[2])])
        if len(data) > 3: # 如果包含强度信息
            intensities.append(float(data[3]))

    return points, intensities


# ========== MAIN_FUCTION =====================================================================

import argparse
parser = argparse.ArgumentParser(description='Process some files')
parser.add_argument('--filename')
parser.add_argument('--ports')
parser.add_argument('--msops')
parser.add_argument('--time_stamp')
parser.add_argument('--dataset_dir')

args = parser.parse_args()
file_name = args.filename
msops = args.msops
ports = args.ports
time_stamp = args.time_stamp
dataset_dir = args.dataset_dir

x_min, x_max = -1000, 1000
y_min, y_max = -1000, 1000
z_min, z_max = -1000, 1000

map_path = dataset_dir + f"/{file_name}/map.pkl"

with open(map_path, 'rb') as file:
    map_dict = pickle.load(file)

with open(dataset_dir + "/Calibration/Roadlidar_to_global.json", "r") as file:
    road_to_global = json.load(file)

for num, sensor in enumerate(map_dict[f'{time_stamp}'][f'{time_stamp}']):
    if str(sensor['port']) == msops:
        pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_{msops}/{sensor['source_time']}.pcd"
        points = filter_point_cloud_by_xyz(pcd_path, x_min, x_max, y_min, y_max, z_min, z_max)
        # point_cloud = o3d.io.read_point_cloud(pcd_path)
        # points = np.asarray(point_cloud.points)
        _, intensities = read_pcd(pcd_path)

    if str(sensor['port']) == ports:
        image_path = dataset_dir + f"/{file_name}" + f"/road_camera/{ports}/{sensor['source_time']}.jpg"
        image = cv2.imread(image_path)

with open(dataset_dir + f'/Calibration/Roadlidar_to_camera/Lidar{msops}_Camera{ports}_T/calib.json') as f:
    calib = json.load(f)

distortion_coeffs = calib['camera']['distortion_coeffs']
intrinsics = calib['camera']['intrinsics']
T = calib['results']['init_T_lidar_camera'] 
transformation_matrix = np.linalg.inv(transform_matrix(*T))
points = np.dot(np.hstack((points, np.ones((points.shape[0], 1)))), transformation_matrix.T)
points = points[:, :3]

fx, fy = intrinsics[0], intrinsics[1] # 焦距
cx, cy = intrinsics[2], intrinsics[3] # 主点
scale = 1000.0 # 假设点云的单位是米，而图片的单位是像素，需要进行相应的缩放

# 根据强度使用色盘来取色
min_intensity = np.min(intensities)
max_intensity = np.max(intensities)
intensity_range = max_intensity - min_intensity
color_map = cv2.applyColorMap(np.uint8(255 * (intensities - min_intensity) / intensity_range), cv2.COLORMAP_JET)

min_intensity = np.min(points[:, 1])
max_intensity = np.max(points[:, 1])
intensity_range = max_intensity - min_intensity
color_map = cv2.applyColorMap(np.uint8(255 * (points[:, 1] - min_intensity) / intensity_range), cv2.COLORMAP_JET)

# 根据相机内参计算视场范围
fov_x = 2 * np.arctan(image.shape[1] / (2 * intrinsics[0]))
fov_y = 2 * np.arctan(image.shape[0] / (2 * intrinsics[1]))

for i, point in enumerate(points):
    # 将点云坐标转换为图像坐标
    u = fx * (point[0] / point[2]) + cx
    v = fy * (point[1] / point[2]) + cy
    x, y, z = point
    if -fov_x / 2 <= np.arctan2(x, z) <= fov_x / 2 and -fov_y / 2 <= np.arctan2(y, z) <= fov_y / 2:
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0] and u != 'nan':
            # print(color)
            # print(color_map[i])
            cv2.circle(image, (int(u), int(v)), radius=1, color=tuple(map(int, color_map[i][0])), thickness=2) # 白色点

# 保存图片
import os
output_path = dataset_dir + f"/{file_name}/visualization/Check_Roadlidar_to_camera/{str(time_stamp)}/{msops}_{ports}_{time_stamp}.jpg"
if not os.path.exists(dataset_dir+ f"/{file_name}/visualization/Check_Roadlidar_to_camera/{str(time_stamp)}/"):
    os.makedirs(dataset_dir+ f"/{file_name}/visualization/Check_Roadlidar_to_camera/{str(time_stamp)}/")
# output_path = f"{msops}_{ports}_{ID}.jpg"
cv2.imwrite(output_path, image)
print("Image saved to:", output_path)