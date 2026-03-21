import numpy as np
import open3d as o3d
import os
import argparse
import math
import pickle
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image
from tqdm import tqdm
from matplotlib.lines import Line2D

# 设置PIL的最大像素限制
Image.MAX_IMAGE_PIXELS = 500000000

# 获取边界框顶点的函数
def get_box_vertices(l, w, h, x, y, z, rot):
    R = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot), np.cos(rot), 0],
        [0, 0, 1]
    ])
    dx, dy, dz = l / 2, w / 2, h / 2
    vertices = np.array([
        [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
        [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]
    ])
    vertices = vertices @ R.T + np.array([x, y, z])
    return vertices

# 计算变换矩阵的函数
def compute_transformation_matrix(tx, ty, tz, roll, pitch, yaw):
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R = R_z @ R_y @ R_x
    t = np.array([tx, ty, tz]).reshape(3, 1)
    T = np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])
    return T

# 获取车辆颜色的函数
def get_vehicle_color(vehicle_type, cmap='tab10'):
    vehicle_types = ["Car", "Motorcycle", "Bicycle", "Pedestrian", "Truck", "Bus", "Trailer", "Van"]
    cmap = plt.get_cmap(cmap)
    color_index = vehicle_types.index(vehicle_type) if vehicle_type in vehicle_types else 0
    return cmap(color_index / len(vehicle_types))

# 裁剪图像的函数
def display_cropped_image(input_path, left, top, right, bottom):
    image = Image.open(input_path)
    cropped_image = image.crop((left, top, image.width - right, image.height - bottom))
    cropped_image.save(input_path)

# 将车辆激光雷达数据转换到路侧坐标系
def Transfer_vehlidar_to_road(file_name, time_stamp, map_dict, pose_dict):
    imu_path = dataset_dir + f"/{file_name}/imu.pkl"
    gps_path = dataset_dir + f"/{file_name}/gps.pkl"
    odom_path = dataset_dir + f"/{file_name}/odom.pkl"
    update_pose_path = dataset_dir + f"/Develop_toolkit/vehicle_pose_update/update_pose/icp_results_{file_name}.json"

    with open(imu_path, 'rb') as file:
        imu_dict = pickle.load(file)
    with open(gps_path, 'rb') as file:
        gps_dict = pickle.load(file)
    with open(odom_path, 'rb') as file:
        odom_dict = pickle.load(file)
    with open(update_pose_path, 'r') as f:
        update_pose = json.load(f)

    T_BLH_ENU = np.array([
        [1.11062385e+05, 4.23280347e+03, -7.94946634e-04, -3.94621043e+06],
        [-4.91081395e+03, 9.57905486e+04, -2.63229088e-03, -1.15269938e+07],
        [-4.90986358e+00, 4.38070505e+00, 9.97741155e-01, -3.98403949e+02]
    ])

    for num, sensor in enumerate(map_dict[f'{time_stamp}'][f'{time_stamp}']):
        if sensor['port'] == 'middle':
            vehlidar_pcd_path = dataset_dir + f"/{file_name}/veh_lidar/middle/{sensor['source_time']}.pcd"

    RTK_x0, RTK_y0, RTK_z0 = 30.88652008, 121.91845964, 16.36
    q0 = (0.0053434877655229324, -0.017198522980084015, 0.35437011649419997, 0.9349318041877119)
    RTK_x = gps_dict[f'{time_stamp}'][f'{time_stamp}']['field.latitude']
    RTK_y = gps_dict[f'{time_stamp}'][f'{time_stamp}']['field.longitude']
    RTK_z = gps_dict[f'{time_stamp}'][f'{time_stamp}']['field.altitude']
    q = (imu_dict[f'{time_stamp}'][f'{time_stamp}']['field.orientation.x'],
         imu_dict[f'{time_stamp}'][f'{time_stamp}']['field.orientation.y'],
         imu_dict[f'{time_stamp}'][f'{time_stamp}']['field.orientation.z'],
         imu_dict[f'{time_stamp}'][f'{time_stamp}']['field.orientation.w'])

    RTK_ENU_0 = np.dot(T_BLH_ENU, np.array([RTK_x0, RTK_y0, RTK_z0, 1]))
    RTK_ENU = np.dot(T_BLH_ENU, np.array([RTK_x, RTK_y, RTK_z, 1]))

    transformation = update_pose[f'{time_stamp}']["final_transformation"]
    transformation_4x4 = np.array([
        [-0.6511579155921936, -0.7583672404289246, -0.02952973172068596, 69.85047912597656],
        [0.7589206695556641, -0.6503510475158691, -0.03289959207177162, 154.08926391601563],
        [0.0057452027685940266, -0.043833568692207336, 0.9990233182907104, 5.200427532196045],
        [0, 0, 0, 1]
    ])
    T_vehlidar_to_RoadRTK = np.dot(np.linalg.inv(transformation_4x4), np.array(transformation).reshape(4, 4))

    vehlidar_pcd = o3d.io.read_point_cloud(vehlidar_pcd_path)
    vehlidar_pcd.transform(T_vehlidar_to_RoadRTK)

    # 加载路侧点云数据
    with open(dataset_dir + "/Calibration/Roadlidar_to_global.json", "r") as file:
        road_to_global = json.load(file)
    for num, sensor in enumerate(map_dict[f'{time_stamp}'][f'{time_stamp}']):
        if sensor['port'] == 6691:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6691/{sensor['source_time']}.pcd"
            pcd1 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd1.transform(road_to_global["6691"])
        if sensor['port'] == 6692:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6692/{sensor['source_time']}.pcd"
            pcd2 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd2.transform(road_to_global["6692"])
        if sensor['port'] == 6699:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6699/{sensor['source_time']}.pcd"
            pcd3 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd3.transform(road_to_global["6699"])
        if sensor['port'] == 6700:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6700/{sensor['source_time']}.pcd"
            pcd4 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd4.transform(road_to_global["6700"])
        if sensor['port'] == 6694:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6694/{sensor['source_time']}.pcd"
            pcd5 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd5.transform(road_to_global["6694"])
        if sensor['port'] == 6701:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6701/{sensor['source_time']}.pcd"
            pcd6 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd6.transform(road_to_global["6701"])
        if sensor['port'] == 6693:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6693/{sensor['source_time']}.pcd"
            pcd7 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd7.transform(road_to_global["6693"])
        if sensor['port'] == 6697:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6697/{sensor['source_time']}.pcd"
            pcd8 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd8.transform(road_to_global["6697"])
        if sensor['port'] == 6695:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6695/{sensor['source_time']}.pcd"
            pcd9 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd9.transform(road_to_global["6695"])
        if sensor['port'] == 6696:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6696/{sensor['source_time']}.pcd"
            pcd10 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd10.transform(road_to_global["6696"])
        if sensor['port'] == 6698:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_6698/{sensor['source_time']}.pcd"
            pcd11 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            pcd11.transform(road_to_global["6698"])
        if sensor['port'] == 21:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_21/{sensor['source_time']}.pcd"
            radar_21 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            radar_21.transform(road_to_global["21"])
        if sensor['port'] == 27:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}/road_lidar/msop_27/{sensor['source_time']}.pcd"
            radar_27 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            radar_27.transform(road_to_global["27"])

    roadlidar_pcd = pcd1 + pcd2 + pcd3 + pcd4 + pcd5 + pcd6 + pcd7 + pcd8 + pcd9 + pcd10 + pcd11

    # 加载标签文件
    json_file = dataset_dir + f"/{file_name}/label_new/road_global_label/{time_stamp}.json"
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = []

    return vehlidar_pcd, roadlidar_pcd, radar_21, radar_27, pcd3, pcd4, pcd6, labels

# 主程序
parser = argparse.ArgumentParser(description='Process some files')
parser.add_argument('--filename', default='20240709_083937_300_1720485599_to_1720485650_51_173')
parser.add_argument('--dataset_dir', default='/home/myData/storage/code/HEAL/dataset/V2XScenes')
parser.add_argument('--plot_view', default='bev')
args = parser.parse_args()

file_name = args.filename
dataset_dir = args.dataset_dir
plot_view = args.plot_view

# 示例参数（用户需根据实际情况调整）
# file_name = '20240709_083937_300_1720485599_to_1720485650_51_173'
# dataset_dir = '/home/myData/storage/code/HEAL/dataset/V2XScenes'
# plot_view = 'bev'  # 或 '3D'

time_path = dataset_dir + f"/{file_name}/Timestamp.pkl"
with open(time_path, 'rb') as file:
    time_list = pickle.load(file)

Plot = True
label_plot = True  # 是否绘制标签，可调整

image_folder = dataset_dir + f'/{file_name}/visualization/Check_Vehicle_to_roadglobal/'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

map_path = dataset_dir + f"/{file_name}/map.pkl"
pose_path = dataset_dir + f"/{file_name}/pose.pkl"

with open(map_path, 'rb') as file:
    map_dict = pickle.load(file)
with open(pose_path, 'rb') as file:
    pose_dict = pickle.load(file)

for num, time in tqdm(enumerate(time_list), total=len(time_list)):
    json_file = dataset_dir + f"/{file_name}/label_new/road_global_label/{time}.json"
    
    if num % 2 == 0 and os.path.exists(json_file) and Plot:
        vehicle_lidar, roadlidar_pcd, radar_21, radar_27, pcd3, pcd4, pcd6, labels = Transfer_vehlidar_to_road(file_name, time, map_dict, pose_dict)
        print(f"===> Time: {time}, Num: {num}, Done: Transfer_vehlidar_to_road")

        if plot_view == "3D":
            fig = plt.figure(figsize=(25, 25))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=40, azim=220)
            fixed_xlim = (-250/1, 100/1)
            fixed_ylim = (0/1, 350/1)
            fixed_zlim = (0, 350/1)
            ax.set_xlim(fixed_xlim)
            ax.set_ylim(fixed_ylim)
            ax.set_zlim(fixed_zlim)
            ax.axis("off")
        else:
            fig = plt.figure(figsize=(30, 10))
            ax = fig.add_subplot()
            fixed_xlim = (-500, 100)
            fixed_ylim = (-50, 150)
            ax.set_xlim(fixed_xlim)
            ax.set_ylim(fixed_ylim)

        vehicle_lidar_points = np.asarray(vehicle_lidar.points)
        roadlidar_points = np.asarray(roadlidar_pcd.points)
        radar_21 = np.asarray(radar_21.points)
        radar_27 = np.asarray(radar_27.points)

        if plot_view == "3D":
            ax.scatter(roadlidar_points[:, 0], roadlidar_points[:, 1], roadlidar_points[:, 2], s=0.005, c='gray', marker='.')
            ax.scatter(radar_21[:, 0], radar_21[:, 1], radar_21[:, 2], s=15.5, c='r', marker='.')
            ax.scatter(radar_27[:, 0], radar_27[:, 1], radar_27[:, 2], s=15.5, c='r', marker='.')
            # ax.scatter(vehicle_lidar_points[:, 0], vehicle_lidar_points[:, 1], vehicle_lidar_points[:, 2], s=0.04, c='darkblue', marker='.')

            if label_plot:
                for label in labels:
                    if '3d_location' in label and '3d_dimensions' in label and 'rotation' in label:
                        location = label['3d_location']
                        dimensions = label['3d_dimensions']
                        rotation = label['rotation']
                        vehicle_type = label['type']
                        x, y, z = location['x'], location['y'], location['z']
                        l, w, h = dimensions['l'], dimensions['w'], dimensions['h']
                        rot = rotation

                        vertices = get_box_vertices(l, w, h, x, y, z, rot)
                        for i in range(4):
                            ax.plot([vertices[i][0], vertices[(i+1)%4][0]],
                                    [vertices[i][1], vertices[(i+1)%4][1]],
                                    [vertices[i][2], vertices[(i+1)%4][2]], color=get_vehicle_color(vehicle_type), lw=0.7)
                            ax.plot([vertices[i+4][0], vertices[(i+1)%4 + 4][0]],
                                    [vertices[i+4][1], vertices[(i+1)%4 + 4][1]],
                                    [vertices[i+4][2], vertices[(i+1)%4 + 4][2]], color=get_vehicle_color(vehicle_type), lw=0.7)
                            ax.plot([vertices[i][0], vertices[i+4][0]],
                                    [vertices[i][1], vertices[i+4][1]],
                                    [vertices[i][2], vertices[i+4][2]], color=get_vehicle_color(vehicle_type), lw=0.7)
                        ax.text(x, y, z, vehicle_type, color=get_vehicle_color(vehicle_type), fontsize=8)

        else:
            theta = np.radians(45)
            rotation_matrix_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                                          [np.sin(theta), np.cos(theta), 0],
                                          [0, 0, 1]])
            roadlidar_points_rot = roadlidar_points @ rotation_matrix_z.T
            radar_21_rot = radar_21 @ rotation_matrix_z.T
            radar_27_rot = radar_27 @ rotation_matrix_z.T
            vehicle_lidar_points_rot = vehicle_lidar_points @ rotation_matrix_z.T

            ax.scatter(roadlidar_points_rot[:, 0], roadlidar_points_rot[:, 1], s=0.1, c='grey', marker='.')
            ax.scatter(radar_21_rot[:, 0], radar_21_rot[:, 1], s=25.5, c='r', marker='.')
            ax.scatter(radar_27_rot[:, 0], radar_27_rot[:, 1], s=25.5, c='r', marker='.')
            ax.scatter(vehicle_lidar_points_rot[:, 0], vehicle_lidar_points_rot[:, 1], s=0.04, c='darkblue', marker='.')

            if label_plot:
                for label in labels:
                    if '3d_location' in label and '3d_dimensions' in label and 'rotation' in label:
                        location = label['3d_location']
                        dimensions = label['3d_dimensions']
                        rotation = label['rotation']
                        vehicle_type = label['type']
                        x, y, z = location['x'], location['y'], location['z']
                        l, w, h = dimensions['l'], dimensions['w'], dimensions['h']
                        rot = rotation + theta

                        center = np.array([x, y, z])
                        center_rotated = rotation_matrix_z @ center
                        x, y, z = center_rotated

                        vertices = get_box_vertices(l, w, h, x, y, z, rot)
                        xy_vertices = vertices[:, :2]
                        for i in range(4):
                            ax.plot([xy_vertices[i][0], xy_vertices[(i+1)%4][0]],
                                    [xy_vertices[i][1], xy_vertices[(i+1)%4][1]], color=get_vehicle_color(vehicle_type), lw=1)
                            ax.plot([xy_vertices[i+4][0], xy_vertices[(i+1)%4 + 4][0]],
                                    [xy_vertices[i+4][1], xy_vertices[(i+1)%4 + 4][1]], color=get_vehicle_color(vehicle_type), lw=1)
                            ax.plot([xy_vertices[i][0], xy_vertices[i+4][0]],
                                    [xy_vertices[i][1], xy_vertices[i+4][1]], color=get_vehicle_color(vehicle_type), lw=1)
                        ax.text(x, y, vehicle_type, color=get_vehicle_color(vehicle_type), fontsize=8)

        legend_elements = [
            Line2D([0], [0], marker='.', color='w', label='Road-side Lidar', markersize=15, markerfacecolor='gray'),
            Line2D([0], [0], marker='.', color='w', label='Vehicle-side Lidar', markersize=15, markerfacecolor='b'),
            Line2D([0], [0], marker='.', color='w', label='Road-side Radar', markersize=15, markerfacecolor='r')
        ]
        ax.text(-120, 75, f'Time: {time}', fontsize=15, color='k')

        image_path = image_folder + f'{time}_{plot_view}.png'
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)
        if plot_view == "3D":
            display_cropped_image(image_path, 0, 6000, 0, 2000)
        plt.close()
        plt.clf()
        print(f"===> Time: {time}, Num: {num}, Done: figure is saved")
