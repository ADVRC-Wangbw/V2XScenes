import numpy as np
import open3d as o3d
import os
import argparse
import math
# from pyproj import Proj, Transformer
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = 500000000  # Set a higher pixel limit
from matplotlib.transforms import Affine2D

def plot_gif(image_folder, output_gif, speed=5, resize_factor=0.5):
    # Get the list of image files and sort them by file name
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()

    frames = []
    
    # Open each image and prepare frames
    for image in tqdm(images, desc="Creating GIF", unit="image"):
        img_path = os.path.join(image_folder, image)
        with Image.open(img_path) as img:
            # Resize the image to make the GIF smaller
            if resize_factor != 1:
                img = img.resize((int(img.width * resize_factor), int(img.height * resize_factor)))
            frames.append(img)
    
    # Save the frames as a GIF with adjusted duration and loop
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=1000 // speed, loop=0)
    
    print(f"GIF file {output_gif} created successfully!")
    
def images_to_gif_ffmpeg(image_folder, output_gif, duration=0.15, max_width=None, max_height=None):
    """
    将图像文件夹中的图片转换为 GIF。

    参数：
        image_folder (str): 包含图片的文件夹路径。
        output_gif (str): 输出 GIF 的文件路径。
        duration (float): 每张图片的持续时间，单位为秒。
        max_width (int, optional): GIF的最大宽度。
        max_height (int, optional): GIF的最大高度。
    """
    # 创建一个临时文本文件，包含所有图片文件的路径和持续时间
    with open("filelist.txt", "w") as filelist:
        images = sorted(
            [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg"))]
        )
        for image in images:
            filelist.write(f"file '{os.path.join(image_folder, image)}'\n")
            filelist.write(f"duration {duration}\n")

    # 构建FFmpeg的缩放和填充参数
    scale_filter = "scale=iw:ih:force_original_aspect_ratio=decrease"
    if max_width and max_height:
        scale_filter = f"scale='min({max_width},iw)':'min({max_height},ih)':force_original_aspect_ratio=decrease"
    
    # 生成 GIF 的 FFmpeg 命令
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'concat',  # 指定输入格式
        '-safe', '0',  # 允许不安全文件路径
        '-i', 'filelist.txt',  # 输入文件列表
        '-vf', scale_filter,  # 调整图像大小
        '-loop', '0',  # 循环播放 GIF
        '-f', 'gif',  # 输出 GIF 格式
        output_gif
    ]

    # 运行 FFmpeg 命令
    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 命令执行失败: {e}")
    finally:
        # 删除临时文件
        if os.path.exists("filelist.txt"):
            os.remove("filelist.txt")

def read_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def get_box_vertices(l, w, h, x, y, z, rot):
    # Define the rotation matrix
    R = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot), np.cos(rot), 0],
        [0, 0, 1]
    ])
    
    # Define the vertices relative to the box center
    dx = l / 2
    dy = w / 2
    dz = h / 2
    vertices = np.array([
        [dx, dy, dz],
        [dx, -dy, dz],
        [-dx, -dy, dz],
        [-dx, dy, dz],
        [dx, dy, -dz],
        [dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx, dy, -dz]
    ])
    
    # Rotate and translate the vertices
    vertices = vertices @ R.T
    vertices += np.array([x, y, z])
    
    return vertices

def compute_transformation_matrix(tx, ty, tz, roll, pitch, yaw):
    """
    根据给定的偏移量和欧拉角计算4x4齐次转换矩阵。

    参数:
    tx (float): x方向的偏移量
    ty (float): y方向的偏移量
    tz (float): z方向的偏移量
    roll (float): 绕x轴的旋转角度（弧度）
    pitch (float): 绕y轴的旋转角度（弧度）
    yaw (float): 绕z轴的旋转角度（弧度）

    返回:
    numpy.ndarray: 4x4 齐次转换矩阵
    """
    
    # 计算旋转矩阵
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 总的旋转矩阵 R
    R = R_z @ R_y @ R_x

    # 位移向量 t
    t = np.array([tx, ty, tz]).reshape(3, 1)

    # 构建 4x4 齐次转换矩阵 T
    T = np.vstack([
        np.hstack([R, t]),
        np.array([0, 0, 0, 1])
    ])

    return T

def Transfer_vehlidar(time_stamp, map_dict):
    
    # tx, ty, tz = 0.915, 0.0, 2.08
    # roll, pitch, yaw = -0.0, 0.0, -1.52
    # T_middle = compute_transformation_matrix(tx, ty, tz, roll, pitch, yaw)

    # tx, ty, tz = 0, 0.615, 2
    # roll, pitch, yaw = -1.300000, 0.0, -0.01535
    # T_left_to_middle = compute_transformation_matrix(tx, ty, tz, roll, pitch, yaw)

    # tx, ty, tz = 0.055, -0.595, 2
    # roll, pitch, yaw = 1.3000000, 0.0000,0.0
    # T_right_to_middle = compute_transformation_matrix(tx, ty, tz, roll, pitch, yaw)
    with open(dataset_dir + "/Calibration/Roadlidar_to_global.json", "r") as file:
        road_to_global = json.load(file)

    tx, ty, tz = 0.0, 0.0, 1.878404
    roll, pitch, yaw = -0.0, 0.0, -1.581749
    T_middle = compute_transformation_matrix(tx, ty, tz, roll, pitch, yaw)

    tx, ty, tz = 0, 0.615, 1.582
    roll, pitch, yaw = -1.23900000, 0.0, 0.0
    T_left_to_middle = compute_transformation_matrix(tx, ty, tz, roll, pitch, yaw)

    tx, ty, tz = 0, -0.595, 1.582
    roll, pitch, yaw = 1.23900000, 0.0000,0.0
    T_right_to_middle = compute_transformation_matrix(tx, ty, tz, roll, pitch, yaw)
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd3 = o3d.geometry.PointCloud()
    for num, sensor in enumerate(map_dict[f'{time_stamp}'][f'{time_stamp}']):
        if sensor['port'] == 'middle':
            vehlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/veh_lidar/middle/{sensor['source_time']}.pcd" 
            pcd1 = o3d.io.read_point_cloud(vehlidar_pcd_path)
            
        if sensor['port'] == 'left':
            vehlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/veh_lidar/left/{sensor['source_time']}.pcd" 
            pcd2 = o3d.io.read_point_cloud(vehlidar_pcd_path)

        if sensor['port'] == 'right':
            vehlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/veh_lidar/right/{sensor['source_time']}.pcd" 
            pcd3 = o3d.io.read_point_cloud(vehlidar_pcd_path)

    # pcd2.transform(np.dot(np.linalg.inv(T_middle),T_left_to_middle))
    # pcd3.transform(np.dot(np.linalg.inv(T_middle),T_right_to_middle))
        # 生成 13 种不同颜色
    colors = np.array([
    [255, 0, 0], # 红色
    [0, 255, 0], # 绿色
    [0, 0, 255], # 蓝色
    [255, 255, 0], # 黄色
    [0, 255, 255], # 青色
    [255, 0, 255], # 品红色
    [255, 165, 0], # 橙色
    [128, 0, 128], # 紫色
    [165, 42, 42], # 棕色
    [255, 192, 203], # 粉色
    [144, 238, 144], # 浅绿色
    [173, 216, 230], # 浅蓝色
    [105, 105, 105] # 深灰色
    ], dtype=np.float32) / 255.0 # Open3D 需要0-1范围内的RGB值
    final_pcd = o3d.geometry.PointCloud()
    pcd2.transform(road_to_global["left"])
    pcd3.transform(road_to_global["right"])
    pcd1.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float32) / 255.0)
    pcd2.colors = o3d.utility.Vector3dVector(np.array([[0, 255, 0]], dtype=np.float32) / 255.0)
    pcd3.colors = o3d.utility.Vector3dVector(np.array([[0, 255, 255]], dtype=np.float32) / 255.0)
    # o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
    # print("T_left_to_middle", np.dot(np.linalg.inv(T_middle),T_left_to_middle))
    # print("T_right_to_middle", np.dot(np.linalg.inv(T_middle),T_right_to_middle))

    final_pcd = pcd1 + pcd2 + pcd3
    # o3d.io.write_point_cloud(dataset_dir + f'/{file_name}/visualization/Check_Vehicle_to_roadglobal/vehlidar_{time}.pcd', final_pcd)

    return final_pcd


def Transfer_roadlidar(time_stamp, map_dict):
    with open(dataset_dir + "/Calibration/Roadlidar_to_global.json", "r") as file:
        road_to_global = json.load(file)
    # print(map_dict[f'{time_stamp}'])
    for num, sensor in enumerate(map_dict[f'{time_stamp}'][f'{time_stamp}']):
        if sensor['port'] == 6691:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6691/{sensor['source_time']}.pcd" 
            pcd1 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T1 = road_to_global["6691"]

        if sensor['port'] == 6692:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6692/{sensor['source_time']}.pcd"
            pcd2 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T2 = road_to_global["6692"]

        if sensor['port'] == 6699:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6699/{sensor['source_time']}.pcd"
            pcd3 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T3 = road_to_global["6699"]
            
        if sensor['port'] == 6700:
            # 读取第4个PCD文件和外参矩阵
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6700/{sensor['source_time']}.pcd"
            pcd4 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T4 = road_to_global["6700"]

        if sensor['port'] == 6694:
            # 读取第5个PCD文件和外参矩阵
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6694/{sensor['source_time']}.pcd"
            pcd5 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T5 = road_to_global["6694"]
            
        if sensor['port'] == 6701:
            # 读取第6个PCD文件和外参矩阵
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6701/{sensor['source_time']}.pcd"
            pcd6 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T6 = road_to_global["6701"]
            
        if sensor['port'] == 6693:
            # 读取第7个PCD文件和外参矩阵
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6693/{sensor['source_time']}.pcd"
            pcd7 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T7 = road_to_global["6693"]
            
        if sensor['port'] == 6697:
            # 读取第8个PCD文件和外参矩阵
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6697/{sensor['source_time']}.pcd"
            pcd8 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T8 = road_to_global["6697"]
            
        if sensor['port'] == 6695:
            # 读取第9个PCD文件和外参矩阵
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6695/{sensor['source_time']}.pcd"
            pcd9 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T9 = road_to_global["6695"]
            
        if sensor['port'] == 6696:
            # 读取第10个PCD文件和外参矩阵
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6696/{sensor['source_time']}.pcd"
            pcd10 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T10 = road_to_global["6696"]
            
        if sensor['port'] == 6698:
            # 读取第11个PCD文件和外参矩阵
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_6698/{sensor['source_time']}.pcd"
            pcd11 = o3d.io.read_point_cloud(roadlidar_pcd_path)
            T11 = road_to_global["6698"]

        if sensor['port'] == 21:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_21/{sensor['source_time']}.pcd"
            T21 = road_to_global["21"]
            radar_21 = o3d.io.read_point_cloud(roadlidar_pcd_path)
        
        if sensor['port'] == 27:
            roadlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/road_lidar/msop_27/{sensor['source_time']}.pcd"
            T27 = road_to_global["27"]
            radar_27 = o3d.io.read_point_cloud(roadlidar_pcd_path)        
            
    # 应用外参矩阵，将点云转换到全局坐标系
    pcd1.transform(T1)
    pcd2.transform(T2)
    pcd3.transform(T3)
    pcd4.transform(T4)
    pcd5.transform(T5)
    pcd6.transform(T6)
    pcd7.transform(T7)
    pcd8.transform(T8)
    pcd9.transform(T9)
    pcd10.transform(T10)
    pcd11.transform(T11)
    radar_21.transform(T21)
    radar_27.transform(T27)
    # 将两个点云拼接在一起
    final_pcd = pcd1 + pcd2 + pcd3 + pcd4 + pcd5 + pcd6 + pcd7 + pcd8 + pcd9 + pcd10 + pcd11 
    return final_pcd, radar_21, radar_27, pcd3, pcd4, pcd6


from scipy.spatial.transform import Rotation as R
import numpy as np

def transform_to_new_origin(position0, quaternion0, position, quaternion):
    """
    Transform a position and quaternion to a new origin defined by another position and quaternion.

    Args:
    - position0: The origin position (array-like of shape (3,))
    - quaternion0: The origin quaternion (array-like of shape (4,))
    - position: The position to transform (array-like of shape (3,))
    - quaternion: The quaternion to transform (array-like of shape (4,))

    Returns:
    - relative_position: The transformed position (array of shape (3,))
    - relative_quaternion: The transformed quaternion (array of shape (4,))
    """
    # Calculate relative position
    relative_position = np.array(position) - np.array(position0)

    # Calculate relative quaternion
    rotation0 = R.from_quat(quaternion0)
    rotation = R.from_quat(quaternion)
    relative_rotation = rotation0.inv()*rotation 
    relative_quaternion = relative_rotation.as_quat()

    return relative_position, relative_quaternion

def compute_relative_pose(p0, q0, p, q):
    """
    计算相对于参考位姿的相对位姿

    参数:
    p0: 参考点位置（3维向量）
    q0: 参考点四元数（4维向量）
    p: 目标点位置（3维向量）
    q: 目标点四元数（4维向量）

    返回:
    相对位置（3维向量）和相对四元数（4维向量）
    """
    # 计算相对位置
    p_relative = p - p0

    # 四元数的逆
    q0_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])

    # 四元数乘法函数
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

    # 计算相对四元数
    q_relative = quaternion_multiply(q0_inv, q)

    return p_relative, q_relative

def quaternion_to_euler(q):
    x, y, z, w = q

    # 计算滚转角 (roll)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # 计算俯仰角 (pitch)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # 计算偏航角 (yaw)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    print(f"Roll: {np.degrees(roll)}, Pitch: {np.degrees(pitch)}, Yaw: {np.degrees(yaw)}")

    return roll, pitch, yaw

def euler_to_rot_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    # print(R)
    return R    
    

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    
    Parameters:
    q (tuple): A tuple (x, y, z, w) representing the quaternion components.
    
    Returns:
    np.ndarray: A 3x3 rotation matrix.
    """
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R
# wgs84 = Proj(proj='latlong', datum='WGS84')
# ecef = Proj(proj='geocent', datum='WGS84')
# transformer = Transformer.from_proj(wgs84, ecef, always_xy=True)

# from pyproj import Proj, transform
# 定义WGS84坐标系（经纬度）
# wgs84 = Proj(init='epsg:4326')

# 定义目标平面坐标系（例如UTM坐标系）
# utm = Proj(init='epsg:32633') # 这里使用UTM Zone 33N作为示例

def gps_to_translation_matrix(lat, lon, alt):
    """
    Convert GPS coordinates into a translation matrix.
    
    Parameters:
    lat (float): Latitude in degrees.
    lon (float): Longitude in degrees.
    alt (float): Altitude in meters.
    
    Returns:
    np.ndarray: A 4x4 translation matrix.
    """

    x = lat
    y = lon
    z = alt
 
    T = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    return T


def combine_matrices(R, T):
    """
    Combine rotation matrix and translation matrix into a single 4x4 matrix.
    
    Parameters:
    R (np.ndarray): A 3x3 rotation matrix.
    T (np.ndarray): A 4x4 translation matrix.
    
    Returns:
    np.ndarray: A combined 4x4 transformation matrix.
    """
    M = np.eye(4)
    M[:3, :3] = R
    M[:, 3] = T[:, 3]
    return M

def Transfer_vehlidar_to_road(file_name, time_stamp, map_dict, pose_dict):

    imu_path = dataset_dir + f"/{file_name}/imu.pkl"
    map_path = dataset_dir + f"/{file_name}/map.pkl"
    gps_path = dataset_dir + f"/{file_name}/gps.pkl"
    odom_path = dataset_dir + f"/{file_name}/odom.pkl"
    update_pose_path = dataset_dir + f"/Develop_toolkit/vehicle_pose_update/update_pose/icp_results_{file_name}.json"

    with open(imu_path, 'rb') as file:
        imu_dict = pickle.load(file)

    with open(gps_path, 'rb') as file:
        gps_dict = pickle.load(file)

    with open(map_path, 'rb') as file:
        map_dict = pickle.load(file)

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
            vehlidar_pcd_path = dataset_dir + f"/{file_name}" + f"/veh_lidar/middle/{sensor['source_time']}.pcd" 

    RTK_x0 = 30.88652008
    RTK_y0 = 121.91845964
    RTK_z0 = 16.36
    q0 = (0.0053434877655229324,-0.017198522980084015,0.35437011649419997,0.9349318041877119) 

    # RTK_x0 = 30.88539257
    # RTK_y0 = 121.91897751
    # RTK_z0 = 21.33
    # q0 = (0.004741316116157475,-0.022811173932213803,-0.36123844231552704,0.9321823630914569) 

    RTK_x = gps_dict[f'{time_stamp}'][f'{time_stamp}']['field.latitude']
    RTK_y = gps_dict[f'{time_stamp}'][f'{time_stamp}']['field.longitude']
    RTK_z = gps_dict[f'{time_stamp}'][f'{time_stamp}']['field.altitude']
    offset = -0.135
    q = (imu_dict[f'{time_stamp}'][f'{time_stamp}']['field.orientation.x'],
        imu_dict[f'{time_stamp}'][f'{time_stamp}']['field.orientation.y'],
        imu_dict[f'{time_stamp}'][f'{time_stamp}']['field.orientation.z'],
        imu_dict[f'{time_stamp}'][f'{time_stamp}']['field.orientation.w']) 
    
    RTK_ENU_0 = np.dot(T_BLH_ENU, np.array([RTK_x0, RTK_y0, RTK_z0, 1]))
    RTK_ENU = np.dot(T_BLH_ENU, np.array([RTK_x, RTK_y, RTK_z, 1]))

    T2 = np.array([
        [1, 0, 0, - RTK_ENU_0[1]],
        [0, 1, 0, - RTK_ENU_0[0]],
        [0, 0, 1, - RTK_ENU_0[2]],
        [0, 0, 0, 1]
        ])
    print(T2)
    # T_vehlidar_to_roadRTK_relative = np.array(
    #     [[-0.652937 ,   0.748208 ,  0.117722  , 23.3918 - RTK_ENU_0[1]],
    #     [ -0.756804  ,  -0.63826 ,  -0.140968   , 50.9489 - RTK_ENU_0[0]],
    #     [-0.0303358   ,  -0.181136,   0.98299   , 7.19614- RTK_ENU_0[2]],
    #     [0   ,        0     ,      0   ,        1]
    #     ])
    
    T_vehlidar_to_roadRTK_relative = np.array([[0.681861 ,   0.725682 ,  0.0919215   ,-0.297959],
            [ -0.731423  ,  0.677992 ,  0.0731325   ,-0.195694],
            [-0.00925113   ,  -0.1171 ,   0.993077   , -10.0937],
            [0   ,        0     ,      0   ,        1]
    ])
    

    RTK_ENU, q = transform_to_new_origin(RTK_ENU_0, q0, RTK_ENU, q)
    roll, pitch, yaw = quaternion_to_euler(q)
    R = euler_to_rot_matrix(roll, pitch, yaw)
    T = gps_to_translation_matrix(RTK_ENU[1], RTK_ENU[0], RTK_ENU[2])
 
    T_GNSS_update = combine_matrices(R, T)
    # T_vehlidar_to_RoadRTK = np.dot(T_GNSS_update, T_vehlidar_to_roadRTK_relative)


    # T_vehlidar_to_RoadRTK = pose_dict[f'{time_stamp}'][f'{time_stamp}'][0]

    transformation = update_pose[f'{time_stamp}']["final_transformation"]
    print(np.array(transformation).reshape(4, 4))
    transformation_4x4 = np.array([
    [-0.6511579155921936,
            -0.7583672404289246,
            -0.02952973172068596,
            69.85047912597656,],
    [0.7589206695556641,
            -0.6503510475158691,
            -0.03289959207177162,
            154.08926391601563,],
    [0.0057452027685940266,
            -0.043833568692207336,
            0.9990233182907104,
            5.200427532196045,],
    [0, 0, 0, 1]
])
    T_vehlidar_to_RoadRTK = np.dot(np.linalg.inv(transformation_4x4), np.array(transformation).reshape(4, 4))

  
    vehlidar_pcd = o3d.io.read_point_cloud(vehlidar_pcd_path)
    
    
    vehlidar_pcd.transform(T_vehlidar_to_RoadRTK)
    # vehlidar_pcd.transform(np.linalg.inv(T2))
    roadlidar_pcd, radar_21, radar_27, pcd3, pcd4, pcd6= Transfer_roadlidar(time_stamp, map_dict)     

    return vehlidar_pcd, roadlidar_pcd, radar_21, radar_27, pcd3, pcd4, pcd6

def read_pcd_file(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('DATA'):
                break
        for line in f:
            data = line.strip().split()
            if len(data) == 0:
                continue
            points.append([float(data[0]), float(data[1]), float(data[2])])
    return np.array(points)

def draw_and_capture_point_cloud(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    return (np.asarray(image) * 255).astype('uint8')

def save_images(images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(output_folder, f"frame_{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

import subprocess

def images_to_video_ffmpeg(image_folder, output_video, duration=0.15, max_width=None, max_height=None):
    """
    将图像文件夹中的图片转换为视频。

    参数：
        image_folder (str): 包含图片的文件夹路径。
        output_video (str): 输出视频的文件路径。
        duration (float): 每张图片的持续时间，单位为秒。
        max_width (int, optional): 视频的最大宽度。
        max_height (int, optional): 视频的最大高度。
    """
    # 创建一个临时文本文件，包含所有图片文件的路径和持续时间
    with open("filelist.txt", "w") as filelist:
        images = sorted(
            [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg"))]
        )
        for image in images:
            filelist.write(f"file '{os.path.join(image_folder, image)}'\n")
            filelist.write(f"duration {duration}\n")

    # 构建FFmpeg的缩放和填充参数
    scale_filter = "scale=iw:ih:force_original_aspect_ratio=decrease"
    if max_width and max_height:
        scale_filter = f"scale='min({max_width},iw)':'min({max_height},ih)':force_original_aspect_ratio=decrease"
    pad_filter = "pad=width=iw+1:height=ih+1:x=(ow-iw)/2:y=(oh-ih)/2:color=black"
    video_filter = f"{scale_filter},{pad_filter}"

    # 构建FFmpeg命令
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'concat',  # 指定输入格式
        '-safe', '0',  # 允许不安全文件路径
        '-i', 'filelist.txt',  # 输入文件列表
        '-vf', video_filter,  # 调整图像大小和填充
        '-vsync', 'vfr',  # 可变帧率
        '-pix_fmt', 'yuv420p',  # 输出视频格式
        output_video
    ]

    # 运行FFmpeg命令
    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 命令执行失败: {e}")
    finally:
        # 删除临时文件
        if os.path.exists("filelist.txt"):
            os.remove("filelist.txt")

def get_vehicle_color(vehicle_type, cmap='tab10'):
    """
    根据交通工具类型返回色盘中的颜色。
    """
    # 定义交通工具类型
    vehicle_types = ["Car", "Motorcycle", "Bicycle", "Pedestrian", "Truck", "Bus", "Trailer", "Van"]
    
    # 获取色盘
    cmap = plt.get_cmap(cmap)  # 使用给定的色盘
    num_colors = len(vehicle_types)
    
    # 为每个交通工具类型分配一个颜色
    color_index = vehicle_types.index(vehicle_type)  # 根据交通工具类型的索引找到对应的颜色
    color = cmap(color_index / num_colors)  # 根据色盘和索引计算对应的颜色
    
    return color

from PIL import Image
def display_cropped_image(input_path, left, top, right, bottom):
    image = Image.open(input_path)
    cropped_image = image.crop((left, top, image.width - right, image.width - bottom))
    cropped_image.save(input_path)

#============ MAIN_FUCTION ================================

parser = argparse.ArgumentParser(description='Process some files')
parser.add_argument('--filename')
parser.add_argument('--dataset_dir')
parser.add_argument('--plot_view')

args = parser.parse_args()
file_name = args.filename
dataset_dir = args.dataset_dir
plot_view = args.plot_view

file_name = '20240709_083937_300_1720485599_to_1720485650_51_173'
dataset_dir = '/home/myData/storage/code/HEAL/dataset/V2XScenes'
plot_view = 'bev'


time_path = dataset_dir + f"/{file_name}/Timestamp.pkl"
with open(time_path, 'rb') as file:
        time_list = pickle.load(file)
Plot = True
label_plot = True 
from tqdm import tqdm

image_folder = dataset_dir + f'/{file_name}/visualization/Check_Vehicle_to_roadglobal_heightfusion++/'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# fixed_xlim = (-290/3, 150/3) # Set your fixed x-axis limits here
# fixed_ylim = (-50/3, 390/3) # Set your fixed y-axis limits here
# fixed_zlim = (0/3, 440/3) # Set your fixed y-axis limits here
 
map_path = dataset_dir + f"/{file_name}/map.pkl"
pose_path = dataset_dir + f"/{file_name}/pose.pkl"

with open(map_path, 'rb') as file:
    map_dict = pickle.load(file)

with open(pose_path, 'rb') as file:
    pose_dict = pickle.load(file)

for num, time in tqdm(enumerate(time_list), total=len(time_list)):
    json_file = dataset_dir + f"/{file_name}/label_new/road_global_label/{time}.json"
    # print(json_file)
    
    if num % 2 == 0 and os.path.exists(json_file) and Plot:
        labels = read_json_file(json_file)
        # Draw and capture point cloud
        '''
        elev（高度角）：设置视角的垂直角度（即从观察者到目标的高度），以度为单位。值通常在 -90 到 +90 度之间。
        elev=0 表示观察者与图形的平面水平对齐。
        elev=90 表示从上方俯视。
        elev=-90 表示从下方查看。
        azim（方位角）：设置视角的水平旋转角度，通常在 0 到 360 度之间。

        azim=0 表示观察者从 x 轴正方向（水平右侧）开始观察。
        azim=90 表示从 y 轴正方向开始观察。
        azim=180 表示从 x 轴负方向（水平左侧）观察。
        azim=270 表示从 y 轴负方向开始观察。
        '''
        if plot_view == "3D":
            fig = plt.figure(figsize=(25, 25))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=30, azim=300)
            fixed_xlim = (-250/3, 100/3) # Set your fixed x-axis limits here # 440
            fixed_ylim = (0/3, 350/3) # Set your fixed y-axis limits here
            fixed_zlim = (0, 350/3) # Set your fixed y-axis limits here
            # fixed_xlim = (90, 100) # Set your fixed x-axis limits here
            # fixed_ylim = (0, 10) # Set your fixed y-axis limits here
            # fixed_zlim = (0, 10) # Set your fixed y-axis limits here
            ax.axis("off")
            ax.view_init(elev=40, azim=220)
            fixed_xlim = (-250/1, 100/1) # Set your fixed x-axis limits here # 440
            fixed_ylim = (0/1, 350/1) # Set your fixed y-axis limits here
            fixed_zlim = (0, 350/1) # Set your fixed y-axis limits here
            ax.set_xlim(fixed_xlim)
            ax.set_ylim(fixed_ylim)
            ax.set_zlim(fixed_zlim)
        else:
            fig = plt.figure(figsize=(30, 10))
            # fixed_xlim = (20, 120) # Set your fixed x-axis limits here
            # fixed_ylim = (-20, 80) # Set your fixed y-axis limits here
            # fixed_zlim = (-0, 100) # Set your fixed y-axis limits here
            # fixed_xlim = (50, 90) # Set your fixed x-axis limits here
            # fixed_ylim = (10, 50) # Set your fixed y-axis limits here
            # fixed_zlim = (-0, 40) # Set your fixed y-axis limits here
            # 画联合标定的图
            # fixed_xlim = (-290, 150) # Set your fixed x-axis limits here
            # fixed_ylim = (-50, 390) # Set your fixed y-axis limits here
            # fixed_zlim = (-0, 440) # Set your fixed y-axis limits here
            ax = fig.add_subplot()
            fixed_xlim = (-500, 100) # Set your fixed x-axis limits here # 440
            fixed_ylim = (-50, 150) # Set your fixed y-axis limits here
            fixed_zlim = (0, 200) # Set your fixed y-axis limits here
            # 画4D雷达的图
            # ax.axis("off")
            # ax.set_box_aspect([2,1,2])
            # ax.set_facecolor('none')
            
            ax.set_xlim(fixed_xlim)
            ax.set_ylim(fixed_ylim)
            
            # 应用旋转到坐标轴

        Transfer_vehlidar(time, map_dict)
        image_path = image_folder + f'{time}_{plot_view}.png'
        vehicle_lidar, roadlidar_pcd, radar_21, radar_27, pcd3, pcd4, pcd6 = Transfer_vehlidar_to_road(f"{file_name}", time, map_dict, pose_dict)
        print(f"===> Time: {time}, Num: {num}, Done: Transfer_vehlidar_to_road")
        if not os.path.exists(dataset_dir + f"/{file_name}/visualization/roadlidar_pcd/"):
            os.makedirs(dataset_dir + f"/{file_name}/visualization/roadlidar_pcd/")
        # o3d.io.write_point_cloud(dataset_dir + f"/{file_name}/visualization/roadlidar_pcd/roadlidar_pcd_{time}.pcd", roadlidar_pcd)
        vehicle_lidar_points = np.asarray(vehicle_lidar.points)
        roadlidar_points = np.asarray(roadlidar_pcd.points)
        pcd3_points = np.asarray(pcd3.points)
        pcd4_points = np.asarray(pcd4.points)
        pcd6_points = np.asarray(pcd6.points)
        radar_21 = np.asarray(radar_21.points)
        radar_27 = np.asarray(radar_27.points)
        
        if plot_view == "3D":
            # ax.scatter(roadlidar_points[:, 0], roadlidar_points[:, 1], roadlidar_points[:, 2], s=0.01, c='gray', marker='.')
            # # ax.scatter(vehicle_lidar_points[:, 0], vehicle_lidar_points[:, 1], s=0.04, c='b', marker='.')
            # ax.scatter(radar_21[:, 0], radar_21[:, 1], radar_21[:, 2], s=1.5, c='r', marker='.')
            # ax.scatter(radar_27[:, 0], radar_27[:, 1], radar_27[:, 2], s=1.5, c='r', marker='.')
            # ax.scatter(pcd4_points[:, 0], pcd4_points[:, 1], s=1, c='darkblue', marker='.')
            # ax.scatter(pcd6_points[:, 0], pcd6_points[:, 1], s=1, c='c', marker='.')
            # ax.scatter(radar_21[:, 0], radar_21[:, 1], s=10.5, c='r', marker='.')

            # 画联合标定的图
            ax.scatter(roadlidar_points[:, 0], roadlidar_points[:, 1], roadlidar_points[:, 2], s=0.005, c='gray', marker='.')
            ax.scatter(radar_21[:, 0], radar_21[:, 1], radar_21[:, 2], s=15.5, c='r', marker='.')
            ax.scatter(radar_27[:, 0], radar_27[:, 1], radar_27[:, 2], s=15.5, c='r', marker='.')
            # ax.scatter(vehicle_lidar_points[:, 0], vehicle_lidar_points[:, 1], vehicle_lidar_points[:, 2], s=0.04, c='darkblue', marker='.')

            # # 画单独4D雷达的图
            # ax.scatter(radar_21[:, 0], radar_21[:, 1], radar_21[:, 2], s=15.5, c='r', marker='.')
            # ax.scatter(radar_27[:, 0], radar_27[:, 1], radar_27[:, 2], s=15.5, c='r', marker='.')

            # def get_vehicle_color(vehicle_type):
            #     """
            #     根据交通工具类型返回指定的颜色。
            #     """
            #     color_map = {
            #         "Car": "blue",        # 蓝色
            #         "Motorcycle": "red",  # 红色
            #         "Bicycle": "green",   # 绿色
            #         "Pedestrian": "gray", # 灰色
            #         "Truck": "purple",    # 紫色
            #         "Bus": "orange",      # 橙色
            #         "Trailer": "brown",   # 棕色
            #         "Van": "cyan"         # 青色
            #     }
                
            #     # 默认返回灰色，如果给定的类型不在字典中
            #     return color_map.get(vehicle_type, "gray")
            if label_plot:
                for i, label in enumerate(labels):
                    # color = 'dodgerblue'
                    # color = 'cornflowerblue'
                    vehicle_types = ["Car", "Motorcycle", "Bicycle", "Pedestrian", "Truck", "Bus", "Trailer", "Van"]
                    if label['type'] in vehicle_types:
                        vehicle_type = label['type']
                        label_color = get_vehicle_color(vehicle_type)
                        length = label['3d_dimensions']['l']
                        width = label['3d_dimensions']['w']
                        height = label['3d_dimensions']['h'] + 0.5
                        center_x = label['3d_location']['x']
                        center_y = label['3d_location']['y']
                        center_z = label['3d_location']['z']
                        angle = label["rotation"]

                        # Calculate the vertices of the 3D box
                        half_length = length / 2
                        half_width = width / 2
                        half_height = height / 2

                        # Calculate the angle in radians
                        angle_rad = angle

                        # Calculate the end coordinates of the arrow
                        arrow_length = 2  # Length of the arrow
                        arrow_end_x = center_x + arrow_length * -np.sin(angle_rad)
                        arrow_end_y = center_y + arrow_length * np.cos(angle_rad)
                        # ax.quiver(center_x, center_y, center_z, arrow_end_x - center_x, arrow_end_y - center_y, 0, color='r', label='Orientation', length=0, normalize=True)

                        # Calculate the 8 vertices of the 3D box
                        vertices = [
                            (center_x + half_length, center_y + half_width, center_z - half_height),
                            (center_x + half_length, center_y - half_width, center_z - half_height),
                            (center_x - half_length, center_y - half_width, center_z - half_height),
                            (center_x - half_length, center_y + half_width, center_z - half_height),
                            (center_x + half_length, center_y + half_width, center_z + half_height),
                            (center_x + half_length, center_y - half_width, center_z + half_height),
                            (center_x - half_length, center_y - half_width, center_z + half_height),
                            (center_x - half_length, center_y + half_width, center_z + half_height),
                        ]

                        # Calculate the rotation matrix based on the angle of orientation
                        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                                                    [0, 0, 1]])

                        # Rotate the vertices
                        rotated_vertices = [np.dot(rotation_matrix, np.array(vertex) - np.array([center_x, center_y, center_z])) + np.array([center_x, center_y, center_z]) for vertex in vertices]

                        # Define the edges of the 3D box
                        edges = [
                            [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4], rotated_vertices[0]],
                            [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6], rotated_vertices[2]],
                            [rotated_vertices[0], rotated_vertices[3], rotated_vertices[7], rotated_vertices[4]],
                            [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]],
                        ]

                        faces = [
                            [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4]],  # Face 1
                            [rotated_vertices[0], rotated_vertices[4], rotated_vertices[7], rotated_vertices[3]],  # Face 2
                            [rotated_vertices[3], rotated_vertices[7], rotated_vertices[6], rotated_vertices[2]],  # Face 3
                            [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]],  # Face 4
                            [rotated_vertices[0], rotated_vertices[3], rotated_vertices[2], rotated_vertices[1]],  # Face 5
                            [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7]],  # Face 6
                        ]

                        # Determine the face pointed to by the arrow
                        arrow_direction = np.array([-np.sin(angle_rad), np.cos(angle_rad), 0])
                        # arrow_direction = np.array([-np.cos(angle_rad), np.sin(angle_rad), 0])
                        arrow_face_index = None
                        for i, face in enumerate(faces):
                            # Normal vector of the face (calculated by cross product)
                            normal_vector = np.cross(np.subtract(face[1], face[0]), np.subtract(face[2], face[0]))
                            
                            # If the arrow points in the opposite direction of the normal vector, it means the arrow points to this face
                            if np.dot(normal_vector, arrow_direction) < 0:
                                arrow_face_index = i + 1
                                # print("i:", i)
                                break

                        # If the face pointed to by the arrow is found, connect the diagonals
                        if arrow_face_index is not None:
                            arrow_face = faces[0]

                            # arrow_face = faces[0]
                            
                            # Calculate the diagonal vertices
                            diagonal_vertices = [
                                [arrow_face[0], arrow_face[2]],
                                [arrow_face[1], arrow_face[3]]
                            ]
                            
                            # Draw the edges of the 3D box
                            for edge in edges:
                                ax.plot([vertex[0] for vertex in edge], 
                                        [vertex[1] for vertex in edge], 
                                        [vertex[2] for vertex in edge], 
                                        color=label_color, lw = 0.7)
                            
                            # Draw the diagonals of the face
                            for diagonal in diagonal_vertices:
                                ax.plot([diagonal[0][0], diagonal[1][0]], [diagonal[0][1], diagonal[1][1]], [diagonal[0][2], diagonal[1][2]], color=label_color, lw = 0.7)
                        else:
                            print("Arrow does not point to any face")
                ax.text(-120, 75, 0, f'Time: {time}', fontsize=15, color='k')

        else:

            # 旋转角度（45度），转换为弧度
            theta = np.radians(45)

            # 绕 z 轴的旋转矩阵
            rotation_matrix_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])

            # 旋转所有点（仅旋转 x 和 y 坐标，z 坐标保持不变）
            roadlidar_points_rot = roadlidar_points @ rotation_matrix_z.T
            radar_21_rot = radar_21 @ rotation_matrix_z.T
            radar_27_rot = radar_27 @ rotation_matrix_z.T
            vehicle_lidar_points_rot = vehicle_lidar_points @ rotation_matrix_z.T

            # 绘制旋转后的点（只展示 x 和 y 坐标，z 坐标不变）
            if num > 30:
                ax.scatter(roadlidar_points_rot[:, 0], roadlidar_points_rot[:, 1], s=0.1, c='gray', marker='.')
                ax.scatter(radar_21_rot[:, 0], radar_21_rot[:, 1], s=25.5, c='r', marker='.')
                ax.scatter(radar_27_rot[:, 0], radar_27_rot[:, 1], s=25.5, c='r', marker='.')
            ax.scatter(vehicle_lidar_points_rot[:, 0], vehicle_lidar_points_rot[:, 1], s=0.04, c='darkblue', marker='.')

            if label_plot:
                for label in labels:
                    vehicle_types = ["Car", "Motorcycle", "Bicycle", "Pedestrian", "Truck", "Bus", "Trailer", "Van"]
                    if label['type'] in vehicle_types:
                        vehicle_type = label['type']
                        label_color = get_vehicle_color(vehicle_type)
                        length = label['3d_dimensions']['l']
                        width = label['3d_dimensions']['w']
                        height = label['3d_dimensions']['h']
                        x_center = label['3d_location']['x']
                        y_center = label['3d_location']['y']
                        z_center = label['3d_location']['z']
                        rotation = label["rotation"]
                        
                        # 获取点云在x和y轴上的范围
                        x_min, x_max = vehicle_lidar_points_rot[:, 0].min(), vehicle_lidar_points_rot[:, 0].max()
                        y_min, y_max = vehicle_lidar_points_rot[:, 1].min(), vehicle_lidar_points_rot[:, 1].max()

                        # 判断目标中心点是否在范围内
                        in_range = (x_min <= x_center <= x_max) and (y_min <= y_center <= y_max)

                        center = np.array([x_center, y_center, z_center])

                        # 旋转物体的中心位置
                        center_rotated = rotation_matrix_z @ center  # 旋转后的物体中心

                        # 物体的旋转矩阵
                        rotation = label["rotation"]

                        # 旋转物体的方向矩阵（旋转矩阵乘以物体的旋转矩阵）
                        rotation_rotated = rotation + theta

                        x_center, y_center, z_center = center_rotated
                    
                        # Get the vertices of the box
                        vertices = get_box_vertices(length, width, height, x_center, y_center, z_center, rotation_rotated)

                        xy_vertices = vertices[:, :2]

                        # Define the edges of the 2D projection
                        edges_2d = [
                            [xy_vertices[0], xy_vertices[1]],
                            [xy_vertices[1], xy_vertices[2]],
                            [xy_vertices[2], xy_vertices[3]],
                            [xy_vertices[3], xy_vertices[0]],
                            [xy_vertices[4], xy_vertices[5]],
                            [xy_vertices[5], xy_vertices[6]],
                            [xy_vertices[6], xy_vertices[7]],
                            [xy_vertices[7], xy_vertices[4]],
                            [xy_vertices[0], xy_vertices[4]],
                            [xy_vertices[1], xy_vertices[5]],
                            [xy_vertices[2], xy_vertices[6]],
                            [xy_vertices[3], xy_vertices[7]]
                        ]

                        # Add each edge to the plot
                        if in_range and num < 30:
                            for edge in edges_2d:
                                ax.plot(*zip(*edge), color=label_color, lw = 1)
                        elif num > 30:
                            for edge in edges_2d:
                                ax.plot(*zip(*edge), color=label_color, lw = 1)
            # ax.text(-120, 75, f'Time: {time}', fontsize=15, color='k')
            ax.axis('off')

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='.', color='w', label='Road-side Lidar', markersize=15, markerfacecolor='gray'),
            Line2D([0], [0], marker='.', color='w', label='Vehicle-side Lidar', markersize=15, markerfacecolor='b'),
            Line2D([0], [0], marker='.', color='w', label='Road-side Radar', markersize=15, markerfacecolor='r')
        ]
        print(f"===> Time: {time}, Num: {num}, Processing: Plot the figure")
        # Add legend
        # ax.legend(handles=legend_elements, loc='best', fontsize='large')
        text = f'Time: {time}'
        
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)
        if plot_view == "3D":
            display_cropped_image(image_path, 0, 6000, 0, 2000)
        plt.close()
        plt.clf()
        
        print(f"===> Time: {time}, Num: {num}, Done: figure is saved")

output_video = dataset_dir + f'/{file_name}/visualization/Check_Vehicle_to_roadglobal/veh_to_roadRTK.mp4'
output_gif = dataset_dir + f'/{file_name}/visualization/Check_Vehicle_to_roadglobal/veh_to_roadRTK.gif'
# images_to_video_ffmpeg(image_folder, output_video)
# images_to_video_ffmpeg(image_folder, output_video, duration=0.2, max_width=1920, max_height=1080)
plot_gif(image_folder, output_gif, speed=5, resize_factor=0.5)
# images_to_gif_ffmpeg(image_folder, output_video, duration=0.15, max_width=None, max_height=None)