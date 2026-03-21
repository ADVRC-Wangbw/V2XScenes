import numpy as np
import open3d as o3d
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from matplotlib.lines import Line2D

# Set PIL image size limit
Image.MAX_IMAGE_PIXELS = 500000000

# ============ Utility Functions ============

def read_json_file(filename):
    """Read JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def get_box_vertices(l, w, h, x, y, z, rot):
    """Get vertices of a 3D box"""
    R = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot), np.cos(rot), 0],
        [0, 0, 1]
    ])
    
    dx, dy, dz = l/2, w/2, h/2
    vertices = np.array([
        [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
        [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]
    ])
    
    vertices = vertices @ R.T
    vertices += np.array([x, y, z])
    return vertices

def compute_transformation_matrix(tx, ty, tz, roll, pitch, yaw):
    """Compute 4x4 transformation matrix from translation and Euler angles"""
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

    R = R_z @ R_y @ R_x
    t = np.array([tx, ty, tz]).reshape(3, 1)
    
    return np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])

def transform_to_new_origin(position0, quaternion0, position, quaternion):
    """Transform a position and quaternion to a new origin"""
    relative_position = np.array(position) - np.array(position0)
    
    rotation0 = R.from_quat(quaternion0)
    rotation = R.from_quat(quaternion)
    relative_rotation = rotation0.inv() * rotation 
    relative_quaternion = relative_rotation.as_quat()

    return relative_position, relative_quaternion

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles"""
    x, y, z, w = q
    
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    return roll, pitch, yaw

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix"""
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    return R_z @ R_y @ R_x

def translation_matrix_from_gps(lat, lon, alt):
    """Create translation matrix from GPS coordinates"""
    T = np.eye(4)
    T[:3, 3] = [lat, lon, alt]
    return T

def combine_matrices(R, T):
    """Combine rotation and translation matrices"""
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = T[:3, 3]
    return M

def get_vehicle_color(vehicle_type, cmap='tab10'):
    """Return color based on vehicle type"""
    vehicle_types = ["Car", "Motorcycle", "Bicycle", "Pedestrian", "Truck", "Bus", "Trailer", "Van"]
    cmap = plt.get_cmap(cmap)
    color_index = vehicle_types.index(vehicle_type)
    return cmap(color_index / len(vehicle_types))

def crop_image(input_path, left, top, right, bottom):
    """Crop image"""
    image = Image.open(input_path)
    cropped_image = image.crop((left, top, image.width - right, image.height - bottom))
    cropped_image.save(input_path)

# ============ Main Data Processing Functions ============

def transfer_road_lidar(timestamp, map_dict, config):
    """Transfer road lidar point clouds to global coordinates"""
    with open(config["road_to_global_path"], "r") as file:
        road_to_global = json.load(file)
    
    # Use dictionary with port as key
    pcd_dict = {}
    for sensor in map_dict[f'{timestamp}'][f'{timestamp}']:
        port = sensor['port']
        if str(port) in config["road_lidar_ports"]:
            
            pcd_path = os.path.join(config["dataset_dir"], config["file_name"], 
                                    f"road_lidar/msop_{port}/{sensor['source_time']}.pcd")
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd.transform(road_to_global[str(port)])
            pcd_dict[port] = pcd
    
    # Combine all point clouds into final_pcd
    if pcd_dict:
        final_pcd = o3d.geometry.PointCloud()
        for pcd in pcd_dict.values():
            final_pcd += pcd
    else:
        final_pcd = o3d.geometry.PointCloud()
    
    return final_pcd, pcd_dict

def transfer_vehicle_lidar_to_road(timestamp, map_dict, pose_dict, config):
    """Transfer vehicle lidar to road coordinates"""
    # Load data
    with open(config["imu_path"], 'rb') as f:
        imu_dict = pickle.load(f)
    with open(config["gps_path"], 'rb') as f:
        gps_dict = pickle.load(f)
    with open(config["update_pose_path"], 'r') as f:
        update_pose = json.load(f)
    
    # Get vehicle lidar path
    vehicle_pcd_path = None
    for sensor in map_dict[f'{timestamp}'][f'{timestamp}']:
        if sensor['port'] == 'middle':
            vehicle_pcd_path = os.path.join(config["dataset_dir"], config["file_name"],
                                           f"veh_lidar/middle/{sensor['source_time']}.pcd")
            break
    
    # Reference point
    RTK_x0, RTK_y0, RTK_z0 = config["reference_gps"]
    q0 = config["reference_quaternion"]
    
    # Current GPS and IMU
    RTK_x = gps_dict[f'{timestamp}'][f'{timestamp}']['field.latitude']
    RTK_y = gps_dict[f'{timestamp}'][f'{timestamp}']['field.longitude']
    RTK_z = gps_dict[f'{timestamp}'][f'{timestamp}']['field.altitude']
    q = (imu_dict[f'{timestamp}'][f'{timestamp}']['field.orientation.x'],
         imu_dict[f'{timestamp}'][f'{timestamp}']['field.orientation.y'],
         imu_dict[f'{timestamp}'][f'{timestamp}']['field.orientation.z'],
         imu_dict[f'{timestamp}'][f'{timestamp}']['field.orientation.w'])
    
    # Transform to ENU
    RTK_ENU_0 = np.dot(config["T_BLH_ENU"], np.array([RTK_x0, RTK_y0, RTK_z0, 1]))
    RTK_ENU = np.dot(config["T_BLH_ENU"], np.array([RTK_x, RTK_y, RTK_z, 1]))
    
    # Get relative transform
    RTK_ENU, q = transform_to_new_origin(RTK_ENU_0, q0, RTK_ENU, q)
    roll, pitch, yaw = quaternion_to_euler(q)
    R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
    T_mat = translation_matrix_from_gps(RTK_ENU[1], RTK_ENU[0], RTK_ENU[2])
    T_GNSS_update = combine_matrices(R_mat, T_mat)
    
    # Vehicle to road transformation
    transformation = update_pose[f'{timestamp}']["final_transformation"]
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
    T_vehlidar_to_road = np.dot(np.linalg.inv(transformation_4x4), np.array(transformation).reshape(4, 4))
    
    # Transform vehicle lidar
    vehicle_pcd = o3d.io.read_point_cloud(vehicle_pcd_path)
    vehicle_pcd.transform(T_vehlidar_to_road)
    
    # Get road lidar
    road_pcd, pcd_dict = transfer_road_lidar(timestamp, map_dict, config)
    
    return vehicle_pcd, road_pcd, pcd_dict

def plot_3d_view(ax, pcd_dict, road_points, vehicle_points, radar_21, radar_27, labels, config, timestamp):
    """Plot 3D view"""
    # Set view limits
    ax.set_xlim(config["fixed_xlim_3d"])
    ax.set_ylim(config["fixed_ylim_3d"])
    ax.set_zlim(config["fixed_zlim_3d"])
    ax.view_init(elev=config["elevation"], azim=config["azimuth"])
    ax.axis("off")
    
    # Plot point clouds
    if road_points is not None and len(road_points) > 0:
        ax.scatter(road_points[:, 0], road_points[:, 1], road_points[:, 2], 
                  s=config["road_lidar_size"], c='gray', marker='.', alpha=0.5)
    if radar_21 is not None and len(radar_21) > 0:
        ax.scatter(radar_21[:, 0], radar_21[:, 1], radar_21[:, 2], 
                  s=config["radar_size"], c='r', marker='.')
    if radar_27 is not None and len(radar_27) > 0:
        ax.scatter(radar_27[:, 0], radar_27[:, 1], radar_27[:, 2], 
                  s=config["radar_size"], c='r', marker='.')
    
    # Plot labels if enabled
    if config["plot_labels"] and labels:
        plot_3d_labels(ax, labels)

def plot_3d_labels(ax, labels):
    """Plot 3D bounding boxes for labels"""
    for label in labels:
        vehicle_types = ["Car", "Motorcycle", "Bicycle", "Pedestrian", "Truck", "Bus", "Trailer", "Van"]
        if label['type'] in vehicle_types:
            color = get_vehicle_color(label['type'])
            
            l = label['3d_dimensions']['l']
            w = label['3d_dimensions']['w']
            h = label['3d_dimensions']['h'] + 0.5
            x = label['3d_location']['x']
            y = label['3d_location']['y']
            z = label['3d_location']['z']
            angle = label["rotation"]
            
            vertices = get_box_vertices(l, w, h, x, y, z, angle)
            
            # Define edges
            edges = [
                [0,1], [1,2], [2,3], [3,0],  # bottom
                [4,5], [5,6], [6,7], [7,4],  # top
                [0,4], [1,5], [2,6], [3,7]   # vertical
            ]
            
            for edge in edges:
                ax.plot([vertices[edge[0], 0], vertices[edge[1], 0]],
                       [vertices[edge[0], 1], vertices[edge[1], 1]],
                       [vertices[edge[0], 2], vertices[edge[1], 2]],
                       color=color, lw=0.7)

def plot_bev_view(ax, pcd_dict, road_points, vehicle_points, radar_21_points, 
                             radar_27_points, labels, config, timestamp):
    """Simplified BEV plotting with loop"""
    theta = np.radians(config["bev_rotation_angle"])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    
    # Color mapping for different ports
    # color_map = {
    #     21: 'darkred', 27: 'darkred',
    #     6691: 'gray', 
    #     6692: 'gray',
    #     6693: 'purple', 6694: 'gray',
    #     6695: 'purple', 6696: 'gray',
    #     6697: 'purple', 6698: 'gray',
    #     6699: 'pink', 6700: 'gray',
    #     6701: 'pink'
    # }
    
    color_map = {
        21: 'red', 27: 'red',
        6691: 'gray', 
        6692: 'gray',
        6693: 'purple', 6694: 'gray',
        6695: 'purple', 6696: 'gray',
        6697: 'purple', 6698: 'gray',
        6699: 'lightpink', 6700: 'gray',
        6701: 'lightpink'
    }
    
    # Plot all road sensors
    for port, pcd in pcd_dict.items():
        points = np.asarray(pcd.points)
        if len(points) > 0:
            points_rot = points @ rotation_matrix.T
            color = color_map.get(port, 'gray')
            marker_size = config["radar_size_bev"] if port in [21, 27] else config["road_lidar_size_bev"]
            ax.scatter(points_rot[:, 0], points_rot[:, 1], 
                      s=marker_size, c=color, marker='.', alpha=0.5, label=f'Port {port}')
    
    # Plot vehicle lidar
    vehicle_points = None
    if vehicle_points is not None and len(vehicle_points) > 0:
        vehicle_points_rot = vehicle_points @ rotation_matrix.T
        ax.scatter(vehicle_points_rot[:, 0], vehicle_points_rot[:, 1], 
                  s=config["vehicle_lidar_size_bev"], c='darkblue', marker='.', alpha=0.7)
    
    ax.set_xlim(config["fixed_xlim_bev"])
    ax.set_ylim(config["fixed_ylim_bev"])
    ax.axis('off')
    
    # if config["plot_labels"] and labels:
    #     plot_bev_labels(ax, labels, rotation_matrix)

def plot_bev_labels(ax, labels, rotation_matrix):
    """Plot BEV bounding boxes for labels"""
    for label in labels:
        types = ["Car", "Motorcycle", "Bicycle", "Pedestrian", "Truck", "Bus", "Trailer", "Van"]
        if label['type'] in types:
            color = get_vehicle_color(label['type'])
            
            l = label['3d_dimensions']['l']
            w = label['3d_dimensions']['w']
            h = label['3d_dimensions']['h']
            x = label['3d_location']['x']
            y = label['3d_location']['y']
            z = label['3d_location']['z']
            rotation = label["rotation"]
            
            # Rotate center
            center = np.array([x, y, z])
            center_rotated = rotation_matrix @ center
            rotation_rotated = rotation + np.radians(45)
            
            vertices = get_box_vertices(l, w, h, center_rotated[0], center_rotated[1], 
                                       center_rotated[2], rotation_rotated)
            xy_vertices = vertices[:, :2]
            
            # Define edges for 2D projection
            edges_2d = [
                [xy_vertices[0], xy_vertices[1]], [xy_vertices[1], xy_vertices[2]],
                [xy_vertices[2], xy_vertices[3]], [xy_vertices[3], xy_vertices[0]],
                [xy_vertices[4], xy_vertices[5]], [xy_vertices[5], xy_vertices[6]],
                [xy_vertices[6], xy_vertices[7]], [xy_vertices[7], xy_vertices[4]],
                [xy_vertices[0], xy_vertices[4]], [xy_vertices[1], xy_vertices[5]],
                [xy_vertices[2], xy_vertices[6]], [xy_vertices[3], xy_vertices[7]]
            ]
            
            for edge in edges_2d:
                ax.plot(*zip(*edge), color=color, lw=1)

# ============ Main Function ============

def main():
    # ============ CONFIGURATION - EDIT THESE VALUES ============
    
    # Dataset parameters
    file_name = '20240709_083937_300_1720485599_to_1720485650_51_173'
    dataset_dir = '/home/myData/storage/code/HEAL/dataset/V2XScenes'
    plot_view = 'bev'  # 'bev' or '3D'
    
    # Paths
    time_path = os.path.join(dataset_dir, file_name, "Timestamp.pkl")
    map_path = os.path.join(dataset_dir, file_name, "map.pkl")
    pose_path = os.path.join(dataset_dir, file_name, "pose.pkl")
    imu_path = os.path.join(dataset_dir, file_name, "imu.pkl")
    gps_path = os.path.join(dataset_dir, file_name, "gps.pkl")
    odom_path = os.path.join(dataset_dir, file_name, "odom.pkl")
    update_pose_path = os.path.join(dataset_dir, "Develop_toolkit/vehicle_pose_update/update_pose", 
                                    f"icp_results_{file_name}.json")
    road_to_global_path = os.path.join(dataset_dir, "Calibration/Roadlidar_to_global.json")
    
    # Output paths
    image_folder = os.path.join(dataset_dir, file_name, "visualization/Check_Vehicle_to_roadglobal_heightfusion++/")
    
    # Road lidar ports
    road_lidar_ports = ['6691', '6692', '6699', '6700', '6694', '6701', '6693', '6697', '6695', '6696', '6698', '21', '27']
    
    # Reference GPS and quaternion
    reference_gps = (30.88652008, 121.91845964, 16.36)
    reference_quaternion = (0.0053434877655229324, -0.017198522980084015, 
                             0.35437011649419997, 0.9349318041877119)
    
    # Transformation matrix
    T_BLH_ENU = np.array([
        [1.11062385e+05, 4.23280347e+03, -7.94946634e-04, -3.94621043e+06],
        [-4.91081395e+03, 9.57905486e+04, -2.63229088e-03, -1.15269938e+07],
        [-4.90986358e+00, 4.38070505e+00, 9.97741155e-01, -3.98403949e+02]
    ])
    
    # Plot settings
    plot_labels = True
    frame_skip = 2  # Process every 2nd frame
    
    # 3D view settings
    elevation = 40
    azimuth = 220
    fixed_xlim_3d = (-250, 100)
    fixed_ylim_3d = (0, 350)
    fixed_zlim_3d = (0, 350)
    road_lidar_size = 0.005
    radar_size = 15.5
    
    # BEV settings
    bev_rotation_angle = 45
    fixed_xlim_bev = (-500, 100)
    fixed_ylim_bev = (-50, 150)
    road_lidar_size_bev = 0.1
    radar_size_bev = 25.5
    vehicle_lidar_size_bev = 0.04
    
    # Create config dictionary
    config = {
        "file_name": file_name,
        "dataset_dir": dataset_dir,
        "plot_view": plot_view,
        "time_path": time_path,
        "map_path": map_path,
        "pose_path": pose_path,
        "imu_path": imu_path,
        "gps_path": gps_path,
        "odom_path": odom_path,
        "update_pose_path": update_pose_path,
        "road_to_global_path": road_to_global_path,
        "image_folder": image_folder,
        "road_lidar_ports": road_lidar_ports,
        "reference_gps": reference_gps,
        "reference_quaternion": reference_quaternion,
        "T_BLH_ENU": T_BLH_ENU,
        "plot_labels": plot_labels,
        "frame_skip": frame_skip,
        "elevation": elevation,
        "azimuth": azimuth,
        "fixed_xlim_3d": fixed_xlim_3d,
        "fixed_ylim_3d": fixed_ylim_3d,
        "fixed_zlim_3d": fixed_zlim_3d,
        "road_lidar_size": road_lidar_size,
        "radar_size": radar_size,
        "bev_rotation_angle": bev_rotation_angle,
        "fixed_xlim_bev": fixed_xlim_bev,
        "fixed_ylim_bev": fixed_ylim_bev,
        "road_lidar_size_bev": road_lidar_size_bev,
        "radar_size_bev": radar_size_bev,
        "vehicle_lidar_size_bev": vehicle_lidar_size_bev
    }
    
    # ============ END OF CONFIGURATION ============
    
    # Create output directory
    os.makedirs(image_folder, exist_ok=True)
    
    # Load timestamp list
    with open(time_path, 'rb') as file:
        time_list = pickle.load(file)
    
    # Load map and pose dictionaries
    with open(map_path, 'rb') as file:
        map_dict = pickle.load(file)
    
    with open(pose_path, 'rb') as file:
        pose_dict = pickle.load(file)
    
    # Process each timestamp
    for num, timestamp in tqdm(enumerate(time_list), total=len(time_list)):
        json_file = os.path.join(dataset_dir, file_name, 
                                 f"label_new/road_global_label/{timestamp}.json")
        
        if num % frame_skip == 0 and os.path.exists(json_file):
            labels = read_json_file(json_file)
            
            # Create figure
            if plot_view == "3D":
                fig = plt.figure(figsize=(25, 25))
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig = plt.figure(figsize=(30, 10))
                ax = fig.add_subplot()
            
            # Transfer point clouds
            vehicle_pcd, road_pcd, pcd_dict = transfer_vehicle_lidar_to_road(
                timestamp, map_dict, pose_dict, config)
            
            print(f"===> Time: {timestamp}, Num: {num}, Done: transfer_vehicle_lidar_to_road")
            
            # Convert to numpy arrays
            vehicle_points = np.asarray(vehicle_pcd.points) if vehicle_pcd else None
            road_points = np.asarray(road_pcd.points) if road_pcd else None
            radar_21_points = np.asarray(pcd_dict.get(21, []).points) if pcd_dict.get(21) else None
            radar_27_points = np.asarray(pcd_dict.get(27, []).points) if pcd_dict.get(27) else None 

            
            # Plot based on view type
            if plot_view == "3D":
                plot_3d_view(ax, pcd_dict, road_points, vehicle_points, radar_21_points, 
                            radar_27_points, labels, config, timestamp)
            else:
                plot_bev_view(ax, pcd_dict, road_points, vehicle_points, radar_21_points, 
                             radar_27_points, labels, config, timestamp)
            
            # Add timestamp text
            # ax.text(0.02, 0.98, f'Time: {timestamp}', transform=ax.transAxes, 
            #        fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save figure
            image_path = os.path.join(image_folder, f'{timestamp}_{plot_view}.png')
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0.1, dpi=300, transparent=False)
            
            # Crop if 3D view
            if plot_view == "3D":
                crop_image(image_path, 0, 6000, 0, 2000)
            
            plt.close()
            plt.clf()
            
            print(f"===> Time: {timestamp}, Num: {num}, Done: figure saved")
    
    print(f"Processing complete! Images saved to {image_folder}")

if __name__ == "__main__":
    main()