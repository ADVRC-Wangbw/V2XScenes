import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import pickle
import shutil
import cv2
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import pickle
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
from tqdm import tqdm
import os
import shutil
# from . import roiaware_pool3d_cuda
import torch

pre_boxes = None
# pre_boxes = np.array([[  17.8590,   75.0607,   -6.3770,    2.0759,    4.5417,    1.4626,
#             3.9917]
#     ])
def create_video_from_images(image_folder, output_video, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size=30):
    """
    将指定文件夹中的图片生成 H.264 编码的视频，并在每张图片上显示文件名。

    :param image_folder: 图片文件夹路径
    :param output_video: 输出视频文件名
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    """
    # 获取所有图片文件的路径并排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # 根据文件名排序

    if not images:
        raise ValueError("指定的文件夹中没有图片。")

    # 读取第一张图片以获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    first_image = Image.open(first_image_path)
    width, height = first_image.size

    # 定义视频编码器（H.264）
    writer = imageio.get_writer(output_video, format='mp4', codec='libx264', fps=10)

    # 加载字体
    font = ImageFont.truetype(font_path, font_size)

    # 初始化进度条
    total_images = len(images)
    with tqdm(total=total_images, desc="生成视频", unit="图片") as pbar:
        # 处理每张图片
        for i, image_file in enumerate(images):
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)

            # 在图片上绘制文本
            draw = ImageDraw.Draw(image)
            text = os.path.splitext(image_file)[0]  # 文件名（不包含扩展名）
            text_position = (10, 10)  # 文本位置
            text_color = (255, 255, 255)  # 文本颜色（白色）

            draw.text(text_position, text, font=font, fill=text_color)

            # 将Pillow图像转换为numpy数组格式
            image_np = np.array(image)

            # 将图像写入视频
            writer.append_data(image_np)

            # 更新进度条
            pbar.update(1)

    # 关闭视频文件
    writer.close()

    print(f"视频已保存为 {output_video}。")


def delete_all_elements(folder_path):
    """删除指定文件夹下的所有文件和子文件夹。"""
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            # 删除文件或子文件夹
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        print(f"已删除 {folder_path} 下的所有元素。")
    else:
        print(f"目录不存在: {folder_path}。")

def read_pcd_file(filename, T, Trans_T):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(filename)
    pcd.transform(T)
    pcd.transform(Trans_T)
    points = np.asarray(pcd.points)

    # points = []
    # with open(filename, 'r') as f:
    #     for line in f:
    #         if line.startswith('DATA'):
    #             break
    #     for line in f:
    #         data = line.strip().split()
    #         if len(data) == 0:
    #             continue
    #         points.append([float(data[0]), float(data[1]), float(data[2])])
    return np.array(points)

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

def read_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def read_bin(bin_file):
    # 读取 .bin 文件中的点云数据
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points

def transform_matrix(x, y, z, qx, qy, qz, qw):
    # 构建旋转矩阵
    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
    print(rotation_matrix)
    # 构建变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [x, y, z]
    return transformation_matrix

def get_bbox_vertices(center, size):
    l, w, h = size / 2
    corners = np.array([[-l, -w, -h],
                        [-l, -w, h],
                        [-l, w, -h],
                        [-l, w, h],
                        [l, -w, -h],
                        [l, -w, h],
                        [l, w, -h],
                        [l, w, h]])
    return corners + center

# 创建3D旋转矩阵
def get_rotation_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

import numpy as np

def rotate_point_around_z_torch(points, theta):
    """
    使用torch绕z轴旋转点云数据
    :param points: (N, 3) 点云数据，N为点数，每行是一个点的 [x, y, z]
    :param theta: 绕z轴的旋转角度（弧度），需要是torch.Tensor类型
    :return: 旋转后的点云数据 (N, 3)
    """
    # 确保theta是Tensor类型
    theta = torch.tensor(theta, dtype=torch.float32)

    rotation_matrix = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    # 使用矩阵乘法进行旋转
    rotated_points = torch.matmul(points, rotation_matrix.T)
    return rotated_points

def is_point_in_box_torch(points, box_min, box_max):
    """
    使用torch判断点云中哪些点在3D框内
    :param points: (N, 3) 点云数据
    :param box_min: (3,) 3D框最小边界 [x_min, y_min, z_min]
    :param box_max: (3,) 3D框最大边界 [x_max, y_max, z_max]
    :return: (N,) 布尔型 tensor，True 表示该点在框内，False 表示不在框内
    """
    # 判断点云是否在3D框内
    inside_box = torch.all((points >= box_min) & (points <= box_max), dim=1)
    return inside_box

def calculate_points_in_3d_box_torch(box_min, box_max, x_center, y_center, z_center, rotation_angle, points):
    """
    计算点云中在3D框内的点数量，使用torch加速计算
    :param label: 包含3D框信息的字典
    :param points: (N, 3) 点云数据，N为点的数量，每行是 [x, y, z]
    :return: 在框内的点数量
    """
    # 将点云数据转换为torch tensor
    points = torch.tensor(points, dtype=torch.float32)

    # 旋转点云数据
    rotated_points = rotate_point_around_z_torch(points - torch.tensor([x_center, y_center, z_center]), rotation_angle)

    # 将旋转后的点云数据恢复到原始坐标系
    rotated_points = rotated_points + torch.tensor([x_center, y_center, z_center])

    # 计算在3D框内的点
    inside_points = is_point_in_box_torch(rotated_points, box_min, box_max)

    # 计算在框内的点数量
    count = inside_points.sum().item()  # 计算True的个数
    return count

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points: (num_points, 3)
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    # print(boxes)
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, is_numpy = check_numpy_to_torch(points)
    boxes, is_numpy = check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    # roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices.numpy() if is_numpy else point_indices

def plot_label_bev(points, json_file, msops, update_json_file, x_min, y_min, z_min, x_max, y_max, z_max):
    # 绘制点云数据
    

    fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot()
    ax.scatter(points[:, 0], points[:, 1], s=5.1, c='b', marker='.')
    if os.path.exists(json_file):
        labels = read_json_file(json_file)
        counts = []
        update_labels = []
        for label in tqdm(labels, desc="Processing labels", unit="label"):
            # 3D box dimensions and location
            length = label['3d_dimensions']['l'] + 1.7
            width = label['3d_dimensions']['w'] + 2.0
            height = label['3d_dimensions']['h'] + 0.5
            x_center = label['3d_location']['x']
            y_center = label['3d_location']['y']
            z_center = label['3d_location']['z']
            rotation = label["rotation"]
            
            # 计算3D框的边界
            half_length = width / 2
            half_width = length / 2
            half_height = height / 2

            box_min = torch.tensor([x_center - half_length, y_center - half_width, z_center - half_height])
            box_max = torch.tensor([x_center + half_length, y_center + half_width, z_center + half_height])

            points = np.array(points)
            count = calculate_points_in_3d_box_torch(box_min, box_max, x_center, y_center, z_center, rotation, points)

            # Get the vertices of the box
            vertices = get_box_vertices(label['3d_dimensions']['l'], label['3d_dimensions']['w'], height, x_center, y_center, z_center, rotation)

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
            for edge in edges_2d:
                ax.plot(*zip(*edge), color='c', lw = 1)

            if count > 1 and msops in ["21", '27']:
                label['sensor_id'] = msops
                counts.append(count)
                for edge in edges_2d:
                    ax.plot(*zip(*edge), color='r', lw = 1)
                update_labels.append(label)
            elif count > 5:
                label['sensor_id'] = msops
                counts.append(count)
                for edge in edges_2d:
                    ax.plot(*zip(*edge), color='r', lw = 1)
                update_labels.append(label)
                

            # # Set plot limits
            # ax.set_xlim([x_center - length, x_center + length])
            # ax.set_ylim([y_center - width, y_center + width])
            # ax.set_zlim(
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.set_xlim([-200, -100])
            # ax.set_ylim([250, 350])
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            # ax.set_zlim([-50, 50])
            # ax.set_ylim([300, 500])
            # ax.set_xlim([-300, -100])

            ax.set_title('Point Cloud')

        print(len(counts), counts)
        print(update_json_file)
        # print(update_labels)
        with open(update_json_file, 'w') as file:
            json.dump(update_labels, file, indent=4)  # Writing with indentation for readability

        if pre_boxes is not None:
            for label in pre_boxes:
                # 3D box dimensions and location
                length = label[4]
                width = label[5]
                height = label[6]
                x_center = label[0]
                y_center = label[1]
                z_center = label[2]
                rotation = label[6]
                
                # Get the vertices of the box
                vertices = get_box_vertices(length, width, height, x_center, y_center, z_center, rotation)

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
                for edge in edges_2d:
                    ax.plot(*zip(*edge), color='g', lw = 1)

                # # Set plot limits
                # ax.set_xlim([x_center - length, x_center + length])
                # ax.set_ylim([y_center - width, y_center + width])
                # ax.set_zlim(
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                ax.set_xlim([-10, 140])
                ax.set_ylim([-100, 50])
                # ax.set_zlim([-50, 50])

                ax.set_title('Point Cloud')


        # 保存图像
        plt.savefig(output_img)
        plt.close()


# MAIN_FUNCTION

import argparse
parser = argparse.ArgumentParser(description='Process some files')
parser.add_argument('--filename')
parser.add_argument('--msops')
parser.add_argument('--sensors')
parser.add_argument('--dataset_dir')

args = parser.parse_args()
file_name = args.filename
msops = args.msops
sensors = args.sensors
dataset_dir = args.dataset_dir

file_name_set = [
    "20240709_083937_300_1720485599_to_1720485650_51_173",
    # "20240709_082823_300_1720485180_to_1720485202_22_73",
    # "20240709_084703_300_1720486033_to_1720486043_10_37",
    # "20240709_172949_300_1720517416_to_1720517437_21_67",
    # "20240709_172949_300_1720517585_to_1720517608_23_65"
]

msops_set = ['21', '27', '6691', '6692', '6693', '6694', '6695', '6696', '6697', '6698', '6699', '6700', '6701']

sensors = 'road_lidar'
# time_stamp = '1720486036.800066'
dataset_dir = "/home/myData/storage/code/HEAL/dataset/V2XScenes" 



for file_name in file_name_set:
    map_path = dataset_dir + f"/{file_name}/map.pkl"
    time_path = dataset_dir + f"/{file_name}/Timestamp.pkl"
    with open(map_path, 'rb') as file:
        map_dict = pickle.load(file)

    with open(time_path, 'rb') as file:
        time_stamps = pickle.load(file)

    with open(dataset_dir + "/Calibration/Roadlidar_to_global.json", "r") as file:
        road_to_global = json.load(file)

    for msops in msops_set:
        if not os.path.exists(dataset_dir+ f"/{file_name}/visualization/plot_label_BEV/{msops}/"):
                    os.makedirs(dataset_dir+ f"/{file_name}/visualization/plot_label_BEV/{msops}/")
        else:
            delete_all_elements(dataset_dir+ f"/{file_name}/visualization/plot_label_BEV/{msops}/")

        if not os.path.exists(dataset_dir+ f"/{file_name}/label/sort_road_lidar_label/{msops}/"):
                    os.makedirs(dataset_dir+ f"/{file_name}/label/sort_road_lidar_label/{msops}/")
        else:
            delete_all_elements(dataset_dir+ f"/{file_name}/label/sort_road_lidar_label/{msops}/")


        if not os.path.exists(dataset_dir+ f"/{file_name}/label/sort_vehicle_lidar_label/{msops}/"):
                    os.makedirs(dataset_dir+ f"/{file_name}/label/sort_vehicle_lidar_label/{msops}/")
        else:
            delete_all_elements(dataset_dir+ f"/{file_name}/label/sort_vehicle_lidar_label/{msops}/")

    init_i = 0

    for i, time_stamp in enumerate(tqdm(time_stamps)):
        for msops in msops_set:
            for num, sensor in enumerate(map_dict[f'{time_stamp}'][f'{time_stamp}']):
                if str(sensor['port']) == str(msops):
                    if sensors == 'veh_lidar':
                        pcd_path = dataset_dir + f"/{file_name}" + f"/{sensors}/{msops}/{sensor['source_time']}.pcd"
                        update_json_file = dataset_dir+ f"/{file_name}/label/sort_vehicle_lidar_label/{msops}/{sensor['source_time']}.json"
                        # json_file = dataset_dir + f"/{file_name}/label/{sensors}/{msops}/{sensor['source_time']}.json"
                        json_file = dataset_dir + f"/{file_name}/label/vehicle_global_label/{sensor['source_time']}.json"
                    else:
                        pcd_path = dataset_dir + f"/{file_name}" + f"/{sensors}/msop_{msops}/{sensor['source_time']}.pcd"
                        update_json_file = dataset_dir+ f"/{file_name}/label/sort_road_lidar_label/{msops}/{sensor['source_time']}.json"
                        # json_file = dataset_dir + f"/{file_name}/label/{sensors}/msop_{msops}/{sensor['source_time']}.json"
                        json_file = dataset_dir + f"/{file_name}/label/road_global_label/{time_stamp}.json"
                    output_img = dataset_dir + f"/{file_name}/visualization/plot_label_BEV/{msops}/{msops}_{sensor['source_time']}_3D.png"
            print(json_file)
            if os.path.exists(json_file) and os.path.exists(pcd_path):
                init_i += init_i
                if sensors == 'veh_lidar':
                    Trans_T = road_to_global[str(msops)]
                    T = np.array([[ 0.999657, -0.00109618, 0.026154, 0], 
                                [ 0, 0.999123, 0.0418757, 0], 
                                [-0.0261769, -0.0418613, 0.99878, 0], 
                                [ 0, 0, 0, 1]])
                    
                    points = read_pcd_file(pcd_path, T, Trans_T)
                else:
                    Trans_T = road_to_global[str(msops)]
                    T = np.array([[ 1, 0, 0, 0], 
                                    [ 0, 1, 0, 0], 
                                    [0, 0, 1, 0], 
                                    [ 0, 0, 0, 1]])
                    points = read_pcd_file(pcd_path, T, Trans_T)
                if init_i == 0:
                    valid_points = points[np.isfinite(points).all(axis=1)]
                    x_min, y_min, z_min = np.min(valid_points, axis=0)
                    x_max, y_max, z_max = np.max(valid_points, axis=0)
                plot_label_bev(points, json_file, msops, update_json_file, x_min, y_min, z_min, x_max, y_max, z_max)

create_video_from_images(dataset_dir+ f"/{file_name}/visualization/plot_label_BEV/{msops}/", dataset_dir+ f"/{file_name}/visualization/plot_label_BEV/{msops}_bev.mp4")
