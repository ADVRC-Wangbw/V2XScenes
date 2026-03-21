"""
V2XScenes to KITTI Format Converter with KITTI-style Visualization

Author: Wangbowen
Date: 2025.05

Description:
This script converts V2XScenes dataset format to KITTI format for 3D object detection.
It handles both vehicle and roadside sensor data: point clouds, images, and labels.
After conversion, it can optionally visualize a sample frame using KITTI-style visualization.

Usage:
1. Configure the USER_CONFIG dictionary with your desired parameters
2. Run the script: python v2xscenes_to_kitti.py
"""

import json
import os
import shutil
import numpy as np
import re
import struct
import math
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# OpenCV for visualization
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not installed. Visualization will be disabled.")

# ===== User Configuration =====
USER_CONFIG = {
    "SELECT_SEQS": [  # Sequences to process
        "20240712_111606_300_1720754373_to_1720754381_8_24",
    ],
    "SELECT_LIDAR_SENSORS": ['6694'],  # Lidar sensors to include // set as "middle" for vehicle-side data
    "BASE_PATH": "/home/myData/storage/code/HEAL/dataset/V2XScenes",  # Base dataset path
    "VISUALIZE": True,  # Whether to visualize a sample frame after conversion
    "VISUALIZE_FRAME_ID": 22,  # Frame index to visualize (0-based)
    "BEV_RESOLUTION": 0.12,  # Resolution for BEV image (meters per pixel)
    "BEV_SIDE_RANGE": (-80.6, 80.6),  # X-axis range in lidar coordinates (left-right)
    "BEV_FWD_RANGE": (-64.8, 64.8),   # Y-axis range in lidar coordinates (forward-backward)
}

Z_AHNGLE_DEG = 0

Z_AHNGLE_DEG_MAPPING = {
   "6700": 90 - 47.294, "6699": 90 - 47.231, "6701": 180 + 44.579, "6698": 90 - 48.462,
    "6696": 90 - 31.458575, "6694": 90 - 27.16 - 14, "6692": 0, "6691": 0,
    "21": 0, "27": 0
}

# ===== Internal Constants =====
CAMERA_NAME_MAP = {
    'back_left_camera': 'BL',
    'back_right_camera': 'BR',
    'front_left_camera': 'FL',
    'front_right_camera': 'FR',
    'left_camera': 'L',
    'right_camera': 'R'
}

POINT_TO_IMAGE_MAPPING = {
    "FUSED": {
        "6700": [66], "6699": [66], "6701": [67], "6698": [75],
        "6696": [64], "6694": [73], "6692": [72], "6691": [71],
        "21": [67], "27": [71]
    },
    "UNFUSED": {
        "6700": [66, 76], "6699": [66], "6701": [67], "6698": [65, 75],
        "6696": [64, 74], "6694": [63, 73], "6692": [62, 72], 
        "6691": [61, 71], "21": [67], "27": [61]
    },
    "VEH": {
        'middle': ['back_left_camera', 'back_right_camera', 
                  'front_left_camera', 'front_right_camera',
                  'left_camera', 'right_camera']
    }
}

COLOR_MAP = {
    "Car": (0, 255, 0), 
    "Bus": (0, 255, 255), 
    "Pedestrian": (255, 255, 0), 
    "Cyclist": (0, 0, 255), 
    "Van": (0, 255, 255), 
    "Truck": (0, 255, 255)
}

# ===== KITTI-style Visualization Functions (adapted from KITTIDataset) =====
def extract_rotation_angles(matrix, angle_type='roll_x', in_degrees=False):
    """
    Extract rotation angles from a 4x4 transformation matrix.
    
    Args:
        matrix: 4x4 numpy array or list, transformation matrix
        angle_type: Type of angle to extract
            - 'roll_x': Rotation around X axis (roll)
            - 'pitch_y': Rotation around Y axis (pitch)
            - 'yaw_z': Rotation around Z axis (yaw)
            - 'yaw_y': Rotation around Y axis (alternative for camera)
            - 'all': Return all angles as dictionary
        in_degrees: If True, return angles in degrees, otherwise radians
    
    Returns:
        float or dict: Requested rotation angle(s)
    """
    # Convert to numpy array
    matrix = np.array(matrix, dtype=np.float64)
    
    # Extract rotation part (3x3)
    if matrix.shape == (4, 4):
        R = matrix[:3, :3]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise ValueError(f"Input must be 3x3 or 4x4 matrix, got shape: {matrix.shape}")
    
    # Calculate angles based on type
    if angle_type == 'roll_x':
        # Roll: rotation around X axis
        # Formula: atan2(R[2,1], R[2,2])
        angle = np.arctan2(R[2, 1], R[2, 2])
        
    elif angle_type == 'pitch_y':
        # Pitch: rotation around Y axis
        # Formula: atan2(-R[2,0], sqrt(R[2,1]^2 + R[2,2]^2))
        angle = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        
    elif angle_type == 'yaw_z':
        # Yaw: rotation around Z axis
        # Formula: atan2(R[1,0], R[0,0])
        angle = np.arctan2(R[1, 0], R[0, 0])
        
    elif angle_type == 'yaw_y':
        # Yaw around Y axis (camera coordinate system)
        # Formula: atan2(R[0,2], R[2,2])
        angle = np.arctan2(R[0, 2], R[2, 2])
        
    elif angle_type == 'all':
        # Return all angles
        angles = {
            'roll_x': np.arctan2(R[2, 1], R[2, 2]),
            'pitch_y': np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)),
            'yaw_z': np.arctan2(R[1, 0], R[0, 0]),
            'yaw_y': np.arctan2(R[0, 2], R[2, 2])
        }
        
        # Convert to degrees if requested
        if in_degrees:
            for key in angles:
                angles[key] = np.degrees(angles[key])
        return angles
        
    else:
        raise ValueError(f"Unsupported angle_type: {angle_type}")
    
    # Convert to degrees if requested
    if in_degrees:
        angle = np.degrees(angle)
    
    return angle

def equation_plane(points):
    """Calculate plane equation from 3 points."""
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])

def get_denorm(Tr_velo_to_cam):
    """Calculate denorm vector for ground plane alignment."""
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(Tr_velo_to_cam, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm

def read_bin(path):
    """Read point cloud from KITTI .bin file."""
    points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
    return points[:, :3]

def lidar2camera_projection(image, points, sensor_params):
    """Project lidar points onto camera image."""
    rmat, tvec, K, dist = sensor_params["rmat"], sensor_params["tvec"], sensor_params["K"], sensor_params["dist"]
    
    image = cv2.undistort(image, K, dist)
    
    # Transform points to camera coordinates and project
    points_cloud = points.copy()
    points_cloud = np.concatenate((points_cloud, np.ones((points_cloud.shape[0], 1))), axis=1)
    points_cloud = points_cloud.T
    extrinsic_matrix = np.concatenate([rmat, tvec.reshape(3, 1)], axis=1)
    image_points = np.dot(np.dot(K, extrinsic_matrix), points_cloud).T.reshape([-1, 3])
    image_points[:, 0] = image_points[:, 0] / image_points[:, 2]
    image_points[:, 1] = image_points[:, 1] / image_points[:, 2]
    image_points = image_points[:, :3].astype(np.int32)
    
    coor, depth = image_points[:, :2], image_points[:, 2]
    height, width, _ = image.shape
    kept = (image_points[:, 0] >= 1) & (image_points[:, 0] < width-1) & (
            image_points[:, 1] >= 1) & (image_points[:, 1] < height-1) & (depth > 1) & (depth < 150)
    coor, depth = coor[kept], depth[kept]
    
    # Color points by depth
    color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    for id in range(coor.shape[0]):
        dis = (depth[id] - 1) / 100 * 256
        dis = min(int(dis), 255)
        color = tuple(color_map[dis, 0].astype(np.uint8))
        image = cv2.circle(image, (coor[id][0], coor[id][1]), 2, 
                          (int(color[0]), int(color[1]), int(color[2])), -1)
    return image

def compute_box_3d_camera(dim, location, rotation_y):
    """
    Compute 3D box corners in camera coordinates (standard KITTI format).
    """
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    
    l, w, h = dim[2], dim[1], dim[0]  # length, width, height
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # y is down in camera coordinates
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    
    return corners_3d.transpose(1, 0)

def compute_box_3d_camera_with_denorm(dim, location, rotation_y, denorm):
    """
    Compute 3D box corners with ground plane alignment.
    """
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    
    # Apply ground plane alignment
    denorm = denorm[:3]
    denorm_norm = denorm / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    ori_denorm = np.array([0.0, -1.0, 0.0])
    theta = -1 * math.acos(np.dot(denorm_norm, ori_denorm))
    n_vector = np.cross(denorm, ori_denorm)
    n_vector_norm = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    rotation_matrix, _ = cv2.Rodrigues(theta * n_vector_norm)
    corners_3d = np.dot(rotation_matrix, corners_3d)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    
    return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
    """Project 3D points to 2D image using camera matrix P."""
    pts_3d_homo = np.concatenate(
        [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d

def draw_box_3d(image, corners, c=(0, 255, 0)):
    """Draw 3D box on image."""
    face_idx = [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    for ind_f in [3, 2, 1, 0]:
        f = face_idx[ind_f]
        for j in [0, 1, 2, 3]:
            cv2.line(image, 
                    (int(corners[f[j], 0]), int(corners[f[j], 1])),
                    (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), 
                    c, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, 
                    (int(corners[f[0], 0]), int(corners[f[0], 1])),
                    (int(corners[f[2], 0]), int(corners[f[2], 1])), 
                    c, 1, lineType=cv2.LINE_AA)
            cv2.line(image, 
                    (int(corners[f[1], 0]), int(corners[f[1], 1])),
                    (int(corners[f[3], 0]), int(corners[f[3], 1])), 
                    c, 1, lineType=cv2.LINE_AA)
    return image

def bbox2image_projection(image, annos, P2, use_denorm=False, denorm=None):
    """Project 3D boxes to 2D image and draw them."""
    for anno in annos:
        loc, dim, rot_y = anno["loc"], anno["dim"], anno["rot_y"]
        obj_type = anno["class"]
        
        if use_denorm and denorm is not None:
            box_3d = compute_box_3d_camera_with_denorm(dim, loc, rot_y, denorm)
        else:
            box_3d = compute_box_3d_camera(dim, loc, rot_y)
            
        box_2d = project_to_image(box_3d, P2)
        c = COLOR_MAP.get(obj_type, (0, 255, 0))
        image = draw_box_3d(image, box_2d, c=c)
    return image

class PointCloudFilter:
    """Filter point cloud and generate BEV image."""
    def __init__(self,
                 side_range=(-39.68, 39.68),
                 fwd_range=(0, 69.12),
                 res=0.10):
        self.res = res
        self.side_range = side_range
        self.fwd_range = fwd_range

    def get_pcl_range(self, points):
        """Keep points within side_range and fwd_range."""
        mask = (points[:, 0] > self.fwd_range[0]) & (points[:, 0] < self.fwd_range[1]) & \
               (points[:, 1] > self.side_range[0]) & (points[:, 1] < self.side_range[1])
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return np.array([]), np.array([])
        x_points = points[indices, 0]
        y_points = points[indices, 1]
        return x_points, y_points

    def get_meshgrid(self):
        """Create empty BEV canvas."""
        x_max = 1 + int((self.side_range[1] - self.side_range[0]) / self.res)
        y_max = 1 + int((self.fwd_range[1] - self.fwd_range[0]) / self.res)
        return np.ones([y_max, x_max, 3], dtype=np.uint8) * 255

    def pcl2xy_plane(self, x_points, y_points):
        """Convert lidar coordinates to BEV pixel coordinates."""
        x_img = (-y_points / self.res).astype(np.int32)   # x in image is -y in lidar
        y_img = (-x_points / self.res).astype(np.int32)   # y in image is -x in lidar
        # Shift to make min (0,0)
        x_img -= int(np.floor(self.side_range[0] / self.res))
        y_img += int(np.ceil(self.fwd_range[1] / self.res))
        return x_img, y_img

    def get_bev_image(self, points_cloud, color=(255, 0, 0), radius=1):
        """Generate BEV image from point cloud."""
        x_points, y_points = self.get_pcl_range(points_cloud)
        if len(x_points) == 0:
            return self.get_meshgrid()

        x_img, y_img = self.pcl2xy_plane(x_points, y_points)
        bev_img = self.get_meshgrid()
        
        for i in range(len(x_img)):
            cv2.circle(bev_img, (x_img[i], y_img[i]), radius, color, -1)
        
        return bev_img

def bbox2bev_projection(bev_image, annos, sensor_params, use_denorm=False, denorm=None):
    """Project 3D boxes to BEV image."""
    rmat, tvec = sensor_params["rmat"], sensor_params["tvec"]
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = rmat
    Tr_velo2cam[:3, 3] = tvec
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    
    # Create point cloud filter for coordinate conversion
    pcf = PointCloudFilter(
        side_range=USER_CONFIG["BEV_SIDE_RANGE"],
        fwd_range=USER_CONFIG["BEV_FWD_RANGE"],
        res=USER_CONFIG["BEV_RESOLUTION"]
    )
    
    for anno in annos:
        loc, dim, rot_y = anno["loc"], anno["dim"], anno["rot_y"]
        obj_type = anno["class"]
        
        # Get box in camera coordinates
        if use_denorm and denorm is not None:
            box_3d_camera = compute_box_3d_camera_with_denorm(dim, loc, rot_y, denorm)
        else:
            box_3d_camera = compute_box_3d_camera(dim, loc, rot_y)
        
        # Transform to lidar coordinates
        box_3d_camera_extend = np.concatenate((box_3d_camera, np.ones((box_3d_camera.shape[0], 1))), axis=1)
        box_3d_lidar = np.matmul(Tr_cam2velo, box_3d_camera_extend.T).T[:, :3]

        # Project to BEV image
        x_img, y_img = pcf.pcl2xy_plane(box_3d_lidar[:, 0], box_3d_lidar[:, 1])
        
        # Draw the box (using top face)
        color = COLOR_MAP.get(obj_type, (0, 255, 0))
        edges = [(0,1), (1,2), (2,3), (3,0)]  # top face edges
        for (i, j) in edges:
            cv2.line(bev_image, (x_img[i], y_img[i]), (x_img[j], y_img[j]), color, 2)
        
    return bev_image

# ===== Core Conversion Functions (unchanged) =====

def extract_xy_rotation_with_z(lidar2global, z_angle_deg=None):
    """Extract rotation from lidar2global while preserving X/Y rotations."""
    rot_matrix = lidar2global[:3, :3]
    r = R.from_matrix(rot_matrix)
    roll, pitch, _ = r.as_euler('xyz', degrees=False)
    if z_angle_deg is None:
        final_yaw = 0
    else:
        final_yaw = np.radians(z_angle_deg)
    R_x = R.from_euler('x', roll).as_matrix()
    R_y = R.from_euler('y', pitch).as_matrix()
    R_z = R.from_euler('z', final_yaw).as_matrix()
    R_combined = R_z @ R_y @ R_x
    lidar2ground = np.eye(4)
    lidar2ground[:3, :3] = R_combined
    lidar2ground[:3, 3] = 0
    return lidar2ground

def transform_matrix(x, y, z, qx, qy, qz, qw):
    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = [x, y, z]
    return T

def create_kitti_structure(root_dir, num_frames):
    subdirs = ['training/velodyne', 'training/calib', 'training/image_2',
               'training/label_2', 'ImageSets']
    for subdir in subdirs:
        path = os.path.join(root_dir, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    train_num = int(0.8 * num_frames)
    with open(os.path.join(root_dir, 'ImageSets', 'train.txt'), 'w') as f:
        f.write('\n'.join(f"{i:06d}" for i in range(train_num)))
    with open(os.path.join(root_dir, 'ImageSets', 'val.txt'), 'w') as f:
        f.write('\n'.join(f"{i:06d}" for i in range(train_num, num_frames)))
    with open(os.path.join(root_dir, 'ImageSets', 'test.txt'), 'w') as f:
        f.write('\n'.join(f"{i:06d}" for i in range(train_num, num_frames)))
    with open(os.path.join(root_dir, 'timestamps.txt'), 'w') as f:
        f.write('\n'.join(f"{i:06d}" for i in range(num_frames)))

def transform_position_and_orientation(x_center, y_center, z_center, z_rotation, T, method=1):
    """Transform a 3D position and z-axis rotation using a given transformation matrix.

    Args:
        x_center (float): Original x-coordinate of the center point.
        y_center (float): Original y-coordinate of the center point.
        z_center (float): Original z-coordinate of the center point.
        z_rotation (float): Original rotation angle around z-axis (in radians).
        T (numpy.ndarray): 4x4 transformation matrix (homogeneous coordinates).
        method (int): Method to calculate delta_z_rotation
                     1 - Use T[1,0] and T[0,0] (standard yaw from rotation matrix)
                     2 - Use -T[2,0] and sqrt(T[2,1]**2 + T[2,2]**2) (alternative)
                     3 - Use T[0,2] and T[2,2] (another alternative)

    Returns:
        tuple: Transformed position and new z-rotation angle as 
               (transformed_x, transformed_y, transformed_z, new_z_rotation).
               new_z_rotation is in radians.
    """
    # Convert original center point to homogeneous coordinates
    original_position = np.array([x_center, y_center, z_center, 1.0])

    # Apply transformation to the position
    transformed_position = np.dot(T, original_position)

    # Extract new center coordinates
    transformed_x_center, transformed_y_center, transformed_z_center, _ = transformed_position

    # Calculate new z-rotation angle based on selected method
    if method == 1:
        # Method 1: Standard yaw from rotation matrix
        delta_z_rotation = np.arctan2(T[1, 0], T[0, 0])
    elif method == 2:
        # Method 2: Using -T[2,0] and sqrt(T[2,1]**2 + T[2,2]**2)
        delta_z_rotation = np.arctan2(-T[2, 0], np.sqrt(T[2, 1]**2 + T[2, 2]**2))
    elif method == 3:
        # Method 3: Using T[0,2] and T[2,2]
        delta_z_rotation = np.arctan2(T[0, 2], T[2, 2])
    else:
        raise ValueError("Method must be 1, 2, or 3")

    new_z_rotation = z_rotation + delta_z_rotation  # Combine with original rotation

    return transformed_x_center, transformed_y_center, transformed_z_center, new_z_rotation

def json_to_kitti_label(json_file_path, output_txt_path, 
                        lidar2global,
                        lidar2ground,
                        global2ground,
                        ground2cam,ground_translation2cam):
    
    """Convert JSON annotation to KITTI format"""
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    
    if not json_file_path or not os.path.exists(json_file_path):
        open(output_txt_path, 'w').close()
        return

    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
    except Exception:
        open(output_txt_path, 'w').close()
        return

    if not json_data:
        open(output_txt_path, 'w').close()
        return
    
    kitti_lines = []
    for obj in json_data:
        obj_type = obj.get("type", "Unknown").replace(" ", "_")
        if obj_type in ['Bicycle', 'Motorcycle']:
            obj_type = 'Cyclist'
        dim = obj.get("3d_dimensions", {})
        loc = obj.get("3d_location", {})
        
        original_rotation = obj['rotation']
        
        loc['x'], loc['y'], loc['z'], obj['rotation'] = transform_position_and_orientation(loc['x'], loc['y'], loc['z'], obj['rotation'], global2ground, method=1)

        loc['z'] -= 0.5 * dim.get('h', 0)
    
        loc['x'], loc['y'], loc['z'], obj['rotation'] = transform_position_and_orientation(loc['x'], loc['y'], loc['z'], obj['rotation'], ground2cam, method=1)
        
        obj['rotation'] = original_rotation
        
        roll_rad = extract_rotation_angles(ground_translation2cam, angle_type='pitch_y', in_degrees=False)
        
        roll_rad_abs = - abs(roll_rad)  # Ensure roll is negative (downward tilt)
        
        if roll_rad > 0:
            obj['rotation'] = - (obj['rotation'] + roll_rad_abs)  # Convert to KITTI format (yaw around z-axis)
        else:
            obj['rotation'] = - (obj['rotation'] + roll_rad) + np.pi # Convert to KITTI format (yaw around z-axis)
        
        roll_x_deg = extract_rotation_angles(ground2cam, angle_type='roll_x', in_degrees=True)
        all_angles_deg = extract_rotation_angles(ground2cam, angle_type='all', in_degrees=True)
        print("\nAll angles (degrees):")
        for angle_type, value in all_angles_deg.items():
            print(f"  {angle_type}: {value:.6f}")     
        
        line = (f"{obj_type} 0 0 -1.5 0 0 0 0 "
                f"{dim.get('h', 0):.6f} {dim.get('w', 0):.6f} {dim.get('l', 0):.6f} "
                f"{loc.get('x', 0):.6f} {loc.get('y', 0):.6f} {loc.get('z', 0):.6f} "
                f"{obj.get('rotation', 0):.6f}")
        kitti_lines.append(line)

    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(kitti_lines))

def parse_pcd_header(f):
    while True:
        pos = f.tell()
        line = f.readline().decode('ascii', errors='ignore').strip()
        if line.startswith("DATA"):
            return {
                'data_format': 'binary' if "binary" in line.lower() else 'ascii',
                'data_start': f.tell(),
                'point_size': 16,
                'expected_points': 1000000
            }
        if len(line) == 0 and pos == f.tell():
            break
    raise ValueError("Invalid PCD header")

def fast_ascii_read(f, start_pos, max_points):
    f.seek(start_pos)
    data = f.read().decode('ascii', errors='ignore')
    lines = data.split('\n')[:max_points]
    pts = np.zeros((len(lines), 4), dtype=np.float32)
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 3:
            pts[i, :3] = list(map(float, parts[:3]))
            if len(parts) >= 4:
                pts[i, 3] = float(parts[3])
    return pts

def fast_binary_read(f, start_pos, point_size, expected_points):
    f.seek(start_pos)
    mm = np.memmap(f, dtype=np.uint8, mode='r', offset=start_pos)
    total = len(mm)
    n = min(expected_points, total // point_size)
    if point_size == 16:
        dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('i', 'f4')])
        pts = np.frombuffer(mm[:n*point_size], dtype=dtype)
        return pts.view(np.float32).reshape(-1, 4)
    else:
        return safe_binary_parse(mm, point_size, n)

def safe_binary_parse(data, point_size, num_points):
    pts = np.zeros((num_points, 4), dtype=np.float32)
    for i in range(num_points):
        off = i * point_size
        try:
            pts[i, 0] = struct.unpack('f', data[off:off+4])[0]
            pts[i, 1] = struct.unpack('f', data[off+4:off+8])[0]
            pts[i, 2] = struct.unpack('f', data[off+8:off+12])[0]
            if point_size >= 16:
                pts[i, 3] = struct.unpack('f', data[off+12:off+16])[0]
        except:
            continue
    return pts

def transform_points(points, T):
    """
    Transform point cloud using transformation matrix T
    
    Args:
        points: numpy array of shape (N, 3) or (N, 4) - point cloud data
        T: 4x4 transformation matrix
    
    Returns:
        transformed_points: new array with transformed coordinates, original points unchanged
    """
    # Create homogeneous coordinates (N, 4)
    hom = np.ones((points.shape[0], 4))
    hom[:, :3] = points[:, :3]
    
    # Apply transformation: transformed = hom @ T.T
    # Using T.T because we want to transform points: [x,y,z,1] @ T.T = transformed point
    transformed = hom @ T.T
    
    # Create result array without modifying original points
    if points.shape[1] == 4:  # If points have intensity (x, y, z, intensity)
        result = np.zeros_like(points)
        result[:, :3] = transformed[:, :3]  # Set transformed coordinates
        result[:, 3] = points[:, 3]  # Keep original intensity values
    else:  # If only XYZ coordinates
        result = transformed[:, :3]  # Return only XYZ
    
    return result

import open3d as o3d
import numpy as np
import os

def pcd_to_bin(pcd_file, bin_file, T):
    with open(pcd_file, 'rb') as f:
        hdr = parse_pcd_header(f)
        if hdr['data_format'] == 'ascii':
            points = fast_ascii_read(f, hdr['data_start'], hdr['expected_points'])
        else:
            points = fast_binary_read(f, hdr['data_start'], hdr['point_size'], hdr['expected_points'])
    
    points_original = points.copy()
    points_trans = transform_points(points, T)
    points_trans.astype(np.float32).tofile(bin_file)
    
    # # 直接指定目标文件夹（例如：/home/user/output/）
    # target_dir = "/home/myData/storage/code/HEAL/dataset/V2XScenes/Dataset_convert/point/V2XScenes_to_KITTI_road_6700_1720754373_1720754381_sync24/training/velodyne_pcd/"
    # bin_name = os.path.basename(bin_file)
    # pcd_name = bin_name.replace('.bin', '_with_axes.pcd')
    # output_pcd_file = os.path.join(target_dir, pcd_name)

    # # 确保目标目录存在
    # os.makedirs(target_dir, exist_ok=True)
    
    # # 创建变换后的点云（红色）
    # trans_pcd = o3d.geometry.PointCloud()
    # trans_pcd.points = o3d.utility.Vector3dVector(points_trans[:, :3])
    # trans_colors = np.full((points_trans.shape[0], 3), [1, 0, 0])  # 红色
    # trans_pcd.colors = o3d.utility.Vector3dVector(trans_colors)
    
    # # 创建原始点云（绿色）
    # orig_pcd = o3d.geometry.PointCloud()
    # orig_pcd.points = o3d.utility.Vector3dVector(points_original[:, :3])
    # orig_colors = np.full((points_original.shape[0], 3), [0, 1, 0])  # 绿色
    # orig_pcd.colors = o3d.utility.Vector3dVector(orig_colors)
    
    # # 合并所有点云
    # combined_pcd = trans_pcd + orig_pcd
    
    # # 确保目标目录存在
    # os.makedirs(target_dir, exist_ok=True)
    
    # # 保存为PCD文件
    # o3d.io.write_point_cloud(output_pcd_file, combined_pcd)
    # print(f"点云已保存到: {output_pcd_file}")
    # print(f"  - 红色: 变换后的点云")

def find_closest_json(pcd_name, json_dir, max_diff=0.1):
    pcd_ts = float(os.path.splitext(pcd_name)[0])
    best, best_diff = None, float('inf')
    for jf in os.listdir(json_dir):
        if not jf.endswith('.json'):
            continue
        try:
            jts = float(os.path.splitext(jf)[0])
            diff = abs(jts - pcd_ts)
            if diff < best_diff:
                best_diff = diff
                best = jf
        except ValueError:
            continue
    return os.path.join(json_dir, best) if best_diff <= max_diff else None

def create_calibration_file(frame_id, lidar_sensor, view, base_path, root_dir,
                            image_mapping, lidar2global, lidar2ground, lidar2ground_translation):
    calib = []
    for i, cam_id in enumerate(image_mapping[lidar_sensor]):
        if lidar_sensor == '21':
            calib_path = os.path.join(base_path, f'Calibration/Roadlidar_to_camera/Lidar6701_Camera{cam_id}_T/calib.json')
        elif lidar_sensor == '27':
            calib_path = os.path.join(base_path, f'Calibration/Roadlidar_to_camera/Lidar6691_Camera{cam_id}_T/calib.json')
        elif lidar_sensor == 'middle':
            calib_path = os.path.join(base_path, f'Calibration/Vehiclelidar_to_camera/Lidar_camera{CAMERA_NAME_MAP[cam_id]}_T/calib.json')
        else:
            calib_path = os.path.join(base_path, f'Calibration/Roadlidar_to_camera/Lidar{lidar_sensor}_Camera{cam_id}_T/calib.json')
        with open(calib_path) as f:
            cam_calib = json.load(f)
        fx, fy, cx, cy = cam_calib['camera']['intrinsics']
        for p in range(4):
            calib.append(f"P{p}: {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0")
    calib.append("R0_rect: 1 0 0 0 1 0 0 0 1")
    for i, cam_id in enumerate(image_mapping[lidar_sensor]):
        if lidar_sensor == '21':
            calib_path = os.path.join(base_path, f'Calibration/Roadlidar_to_camera/Lidar6701_Camera{cam_id}_T/calib.json')
        elif lidar_sensor == '27':
            calib_path = os.path.join(base_path, f'Calibration/Roadlidar_to_camera/Lidar6691_Camera{cam_id}_T/calib.json')
        elif lidar_sensor == 'middle':
            calib_path = os.path.join(base_path, f'Calibration/Vehiclelidar_to_camera/Lidar_camera{CAMERA_NAME_MAP[cam_id]}_T/calib.json')
        else:
            calib_path = os.path.join(base_path, f'Calibration/Roadlidar_to_camera/Lidar{lidar_sensor}_Camera{cam_id}_T/calib.json')
        with open(calib_path) as f:
            cam_calib = json.load(f)
        T = cam_calib['results']['init_T_lidar_camera']
        lidar2cam = np.linalg.inv(transform_matrix(*T))
        ground2cam = lidar2cam @ np.linalg.inv(lidar2ground)
        ground_translation2cam = lidar2cam @ np.linalg.inv(lidar2ground_translation)
        Tr = ground2cam[:3, :4]
        calib.append("Tr_velo_to_cam: " + ' '.join(map(str, Tr.flatten())))
    calib.append("Tr_velo_to_global: " + ' '.join(map(str, lidar2global.flatten())))
    out_path = os.path.join(root_dir, 'training/calib', f"{frame_id:06d}.txt")
    with open(out_path, 'w') as f:
        f.write('\n'.join(calib))
    return ground2cam, ground_translation2cam

def process_scene(scene_id, scene_data, lidar_sensor, view, base_path, root_dir,
                 image_mapping, fused_mode, camera_counter_set):
    if view == 'road':
        lidar_src = os.path.join(base_path, scene_id, f'{view}_lidar/msop_{lidar_sensor}')
        json_dir = os.path.join(base_path, f'label_process/V2XScenes_label/{scene_id}/sort_road_lidar_label/{lidar_sensor}/')
        calib_file = os.path.join(base_path, 'Calibration/Roadlidar_to_global.json')
        with open(calib_file) as f:
            road_to_global = json.load(f)
        lidar2global = np.array(road_to_global[lidar_sensor])
        # lidar2ground = extract_xy_rotation_with_z(lidar2global, z_angle_deg=Z_AHNGLE_DEG_MAPPING[USER_CONFIG['SELECT_LIDAR_SENSORS'][0]])
        # global2ground = np.linalg.inv(lidar2ground) @ lidar2global
        t = lidar2global[:3, 3].copy()

        # 创建从全局到平移后坐标系的变换（无旋转）
        global2ground_translation = np.eye(4)
        global2ground_translation[:3, 3] = -t
        lidar2ground_translation = global2ground_translation @ lidar2global
        
        # 创建绕Z轴的旋转矩阵
        theta = Z_AHNGLE_DEG_MAPPING[USER_CONFIG['SELECT_LIDAR_SENSORS'][0]] * np.pi / 180  # 你的自定义角度（弧度）
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 先平移，后旋转
        
        global2ground = R_z @ global2ground_translation
        lidar2ground = global2ground @ lidar2global
        
    else:
        lidar_src = os.path.join(base_path, scene_id, f'{view}_lidar/{lidar_sensor}')
        json_dir = os.path.join(base_path, f'label_process/V2XScenes_label/{scene_id}/veh_global_label/')
        calib_file = os.path.join(base_path, 'Calibration/Roadlidar_to_global.json')
        with open(calib_file) as f:
            road_to_global = json.load(f)
        lidar2global = np.array(road_to_global['middle_to_ground'])
        lidar2ground = global2ground @ lidar2global
        # lidar2ground = extract_xy_rotation_with_z(lidar2global, z_angle_deg=Z_AHNGLE_DEG)
        # global2ground = np.linalg.inv(lidar2ground) @ lidar2global
    velodyne_dir = os.path.join(root_dir, 'training/velodyne')
    img_src_dirs, img_dst_dirs = [], []
    for i, cam_id in enumerate(image_mapping[lidar_sensor]):
        src = os.path.join(base_path, scene_id, f'{view}_camera/{cam_id}')
        dst_dir = os.path.join(root_dir, 'training', f'image_{i+2}')
        os.makedirs(dst_dir, exist_ok=True)
        img_src_dirs.append(src)
        img_dst_dirs.append(dst_dir)
    timestamp_list = []
    for lidar_file in tqdm(sorted(os.listdir(lidar_src)), desc=f"{scene_id}", leave=False):
        frame_id = len(os.listdir(velodyne_dir))
        timestamp_list.append(f"{frame_id:06d}.bin")
        src_pcd = os.path.join(lidar_src, lidar_file)
        dst_bin = os.path.join(velodyne_dir, f"{frame_id:06d}.bin")
        pcd_to_bin(src_pcd, dst_bin, lidar2ground)
        ground2cam, ground_translation2cam = create_calibration_file(frame_id, lidar_sensor, view, base_path,
                                             root_dir, image_mapping, lidar2global, lidar2ground, lidar2ground_translation)
        json_path = find_closest_json(lidar_file, json_dir)
        json_to_kitti_label(json_path,
                            os.path.join(root_dir, 'training/label_2', f"{frame_id:06d}.txt"),
                            lidar2global, lidar2ground, global2ground, ground2cam, ground_translation2cam)
    for k, src_dir in enumerate(img_src_dirs):
        for img_name in tqdm(sorted(os.listdir(src_dir), key=lambda x: int(x.split('.')[0])),
                             desc=f"Images cam{k}", leave=False):
            src = os.path.join(src_dir, img_name)
            new_name = f"{camera_counter_set[k]:06d}.jpg"
            dst = os.path.join(img_dst_dirs[k], new_name)
            shutil.copy(src, dst)
            camera_counter_set[k] += 1
    ts_path = os.path.join(root_dir, 'timestamps.txt')
    with open(ts_path, 'r') as f:
        lines = f.readlines()
    for i, ts in enumerate(timestamp_list):
        if i < len(lines):
            lines[i] = ts + '\n'
    with open(ts_path, 'w') as f:
        f.writelines(lines)

# ===== KITTI-style Visualization Wrapper =====

def kitti_style_visualization(root_dir, frame_id=0, use_denorm=False):
    """
    Visualize a frame using KITTI-style visualization.
    This matches the visualization style of the KITTIDataset code.
    """
    # File paths
    bin_file = os.path.join(root_dir, 'training/velodyne', f"{frame_id:06d}.bin")
    label_file = os.path.join(root_dir, 'training/label_2', f"{frame_id:06d}.txt")
    calib_file = os.path.join(root_dir, 'training/calib', f"{frame_id:06d}.txt")
    image_file = os.path.join(root_dir, 'training/image_2', f"{frame_id:06d}.jpg")
    
    if not all(os.path.exists(f) for f in [bin_file, label_file, calib_file, image_file]):
        print(f"Warning: Frame {frame_id} files not found, skipping visualization.")
        return
    
    print(f"KITTI-style visualization for frame {frame_id:06d}...")
    
    # Load calibration
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    P2 = None
    Tr_velo_to_cam = None
    for line in lines:
        if line.startswith('P2:'):
            values = list(map(float, line.strip().split()[1:]))
            P2 = np.array(values).reshape(3, 4)
        elif line.startswith('Tr_velo_to_cam:'):
            values = list(map(float, line.strip().split()[1:]))
            Tr_velo_to_cam = np.array(values).reshape(3, 4)
    
    if P2 is None or Tr_velo_to_cam is None:
        print("Error: Could not load calibration matrices")
        return
    
    # Load image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Could not load image {image_file}")
        return
    
    # Load point cloud
    points = read_bin(bin_file)
    
    # Prepare sensor parameters
    sensor_params = {
        "rmat": Tr_velo_to_cam[:3, :3],
        "tvec": Tr_velo_to_cam[:3, 3],
        "K": P2[:3, :3],
        "dist": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    
    # Calculate denorm if needed
    denorm = get_denorm(Tr_velo_to_cam) if use_denorm else None
    
    # Project lidar points to image
    image = lidar2camera_projection(image, points, sensor_params)
    
    # Load annotations
    annotations = []
    with open(label_file, 'r') as f:
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 
                      'dh', 'dw', 'dl', 'lx', 'ly', 'lz', 'ry']
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            annotations.append({
                "class": parts[0],
                "dim": [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                "loc": [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z
                "rot_y": float(parts[14])
            })
    
    # Draw 3D boxes on image
    image = bbox2image_projection(image, annotations, P2, use_denorm, denorm)
    
    # Create BEV image
    pcf = PointCloudFilter(
        side_range=USER_CONFIG["BEV_SIDE_RANGE"],
        fwd_range=USER_CONFIG["BEV_FWD_RANGE"],
        res=USER_CONFIG["BEV_RESOLUTION"]
    )
    bev_image = pcf.get_bev_image(points[:, :3], color=(255, 0, 0), radius=2)
    
    # Draw boxes on BEV
    bev_image = bbox2bev_projection(bev_image, annotations, sensor_params, use_denorm, denorm)
    
    # Save results
    out_image = os.path.join(root_dir, f'vis_frame_{frame_id:06d}_kitti.jpg')
    out_bev = os.path.join(root_dir, f'vis_frame_{frame_id:06d}_kitti_bev.jpg')
    
    cv2.imwrite(out_image, image)
    cv2.imwrite(out_bev, bev_image)
    
    print(f"KITTI-style image saved to {out_image}")
    print(f"KITTI-style BEV saved to {out_bev}")

# ===== Main Conversion Function =====

def convert_to_kitti():
    config = USER_CONFIG
    select_seqs = config["SELECT_SEQS"]
    select_lidar_sensors = config["SELECT_LIDAR_SENSORS"]
    base_path = config["BASE_PATH"]
    camera_counter_set = [0, 0, 0, 0, 0, 0]
    
    fused_mode = True
    view = "veh" if select_lidar_sensors == ['middle'] else "road"
    sync_count = len(select_lidar_sensors) * sum(int(seq.split('_')[-1]) for seq in select_seqs)
    timestamps = [t for seq in select_seqs for t in re.findall(r'1720\d+', seq)[:2]]
    sensor_part = "_".join(select_lidar_sensors)
    folder_name = f"V2XScenes_to_KITTI_{view}_{sensor_part}_{timestamps[0]}_{timestamps[-1]}_sync{sync_count}"
    root_dir = os.path.join(base_path, 'Dataset_convert/point', folder_name)
    
    create_kitti_structure(root_dir, sync_count)

    with open(os.path.join(base_path, 'label_process/label_map.json'), 'r') as f:
        labeled_seqs = json.load(f)

    print(f"===== Converting {len(select_seqs)} sequences =====")
    print(f"Mode: FUSED, View: {view.upper()}")
    
    image_mapping = POINT_TO_IMAGE_MAPPING["FUSED"]
    
    for lidar_sensor in tqdm(select_lidar_sensors, desc="Lidar Sensors"):
        for seq in tqdm(labeled_seqs, desc=f"Sequences for {lidar_sensor}", leave=False):
            for scene_id, scene_data in seq.items():
                if scene_id in select_seqs:
                    process_scene(scene_id, scene_data, lidar_sensor, view,
                                base_path, root_dir, image_mapping, fused_mode, camera_counter_set)
    
    if config.get("VISUALIZE", False) and CV2_AVAILABLE:
        frame_id = config.get("VISUALIZE_FRAME_ID", 0)
        kitti_style_visualization(root_dir, frame_id, use_denorm=True)

if __name__ == "__main__":
    convert_to_kitti()