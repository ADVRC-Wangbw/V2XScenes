"""
Utility functions for data processing and visualization in V2X scenarios.
"""

import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def delete_all_elements(folder_path: str) -> None:
    """
    Delete all files and subdirectories in the specified folder.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder to clean
    """
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"Deleted all contents of: {folder_path}")
    else:
        print(f"Directory does not exist: {folder_path}")


def get_box_vertices(l: float, w: float, h: float, 
                     x: float, y: float, z: float, 
                     rot: float) -> np.ndarray:
    """
    Calculate the 8 vertices of a 3D bounding box.
    
    Parameters
    ----------
    l : float
        Length of the box
    w : float
        Width of the box
    h : float
        Height of the box
    x, y, z : float
        Center coordinates of the box
    rot : float
        Rotation angle around Z-axis (yaw) in radians
    
    Returns
    -------
    np.ndarray
        Array of shape (8, 3) containing the 3D vertices coordinates
    """
    # Rotation matrix around Z-axis
    R_mat = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot), np.cos(rot), 0],
        [0, 0, 1]
    ])
    
    # Half dimensions
    dx, dy, dz = l / 2, w / 2, h / 2
    
    # Vertices relative to box center (local coordinates)
    vertices_local = np.array([
        [ dx,  dy,  dz],  # front top right
        [ dx, -dy,  dz],  # front top left
        [-dx, -dy,  dz],  # back top left
        [-dx,  dy,  dz],  # back top right
        [ dx,  dy, -dz],  # front bottom right
        [ dx, -dy, -dz],  # front bottom left
        [-dx, -dy, -dz],  # back bottom left
        [-dx,  dy, -dz],  # back bottom right
    ])
    
    # Rotate and translate vertices to world coordinates
    vertices_world = vertices_local @ R_mat.T + np.array([x, y, z])
    
    return vertices_world

def plot_label_bev(points, points2, labels, name, seq_name, save_dir='./opencood/vis_training_v2xscenes/'):
    """
    Plot BEV (Bird's Eye View) visualization of point clouds and 3D bounding boxes.
    
    Parameters
    ----------
    points : numpy.ndarray
        Point cloud data (N, 3)
    points2 : numpy.ndarray
        Secondary point cloud data (not used in current implementation)
    labels : list or dict
        Bounding box labels, each containing [x, y, z, h, w, l, yaw] or dict format
    name : str
        Output filename
    seq_name : str
        Sequence name for directory organization
    save_dir : str
        Base directory for saving plots
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    
    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], s=0.1, c='darkblue', marker='.')
    
    # Plot bounding boxes
    for label in labels:
        # Extract bounding box parameters
        # Assuming label format: [x, y, z, h, w, l, yaw]
        x_center = label[0]
        y_center = label[1]
        z_center = label[2]
        height = label[3]   # h
        width = label[4]    # w
        length = label[5]   # l
        rotation = label[6] # yaw
        
        # Get 3D box vertices
        vertices = get_box_vertices(length, width, height, x_center, y_center, z_center, rotation)
        
        # Project to 2D (BEV)
        xy_vertices = vertices[:, :2]
        
        # Define edges for 2D projection
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
        
        # Draw each edge
        for edge in edges_2d:
            ax.plot(*zip(*edge), color='r', lw=1)
    
    # Set plot limits (LiDAR range)
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    
    # Set labels and ticks
    ax.set_xlabel('X [m]', fontsize=22)
    ax.set_ylabel('Y [m]', fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    
    # Create save directory
    seq_name_no_ext = os.path.splitext(seq_name)[0]
    save_path = os.path.join(save_dir, seq_name_no_ext)
    os.makedirs(save_path, exist_ok=True)
    
    # Save figure
    plt.savefig(
        os.path.join(save_path, f'{name}.png'),
        bbox_inches='tight',
        pad_inches=0.1,
        dpi=300
    )
    plt.close()
    print(f"Plot saved successfully: {os.path.join(save_path, f'{name}.png')}")

def plot_label_3D(points, points2, labels, name, seq_name, save_dir='./opencood/vis_training_v2xscenes/'):
    """
    Plot 3D visualization of point cloud and 3D bounding boxes.
    
    Parameters
    ----------
    points : numpy.ndarray
        Point cloud data (N, 3)
    points2 : numpy.ndarray
        Secondary point cloud data (not used)
    labels : list
        Bounding box labels, each containing [x, y, z, h, w, l, yaw]
    name : str
        Output filename
    seq_name : str
        Sequence name for directory organization
    save_dir : str
        Base directory for saving plots (default: './opencood/vis_training_v2xscenes/')
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              s=0.04, c='darkblue', marker='.')
    
    # Plot bounding boxes
    for label in labels:
        # Extract bounding box parameters
        x_center = label[0]
        y_center = label[1]
        z_center = label[2]
        height = label[3]   # h
        width = label[4]    # w
        length = label[5]   # l
        yaw = label[6]      # rotation angle in radians
        
        # Calculate half dimensions
        half_length = length / 2
        half_width = width / 2
        half_height = height / 2
        
        # Calculate unrotated vertices
        vertices_local = np.array([
            [ half_length,  half_width, -half_height],
            [ half_length, -half_width, -half_height],
            [-half_length, -half_width, -half_height],
            [-half_length,  half_width, -half_height],
            [ half_length,  half_width,  half_height],
            [ half_length, -half_width,  half_height],
            [-half_length, -half_width,  half_height],
            [-half_length,  half_width,  half_height],
        ])
        
        # Rotation matrix around Z-axis
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Rotate and translate vertices
        vertices = np.dot(vertices_local, rot_matrix.T) + np.array([x_center, y_center, z_center])
        
        # Define edges of the 3D box
        edges = [
            [0, 1, 5, 4, 0],  # front face
            [2, 3, 7, 6, 2],  # back face
            [0, 3, 7, 4],      # left face
            [1, 2, 6, 5],      # right face
        ]
        
        # Define faces for orientation detection
        faces = [
            [0, 1, 5, 4],  # front
            [0, 4, 7, 3],  # left
            [3, 7, 6, 2],  # back
            [1, 2, 6, 5],  # right
            [0, 3, 2, 1],  # bottom
            [4, 5, 6, 7],  # top
        ]
        
        # Draw edges
        for edge in edges:
            edge_vertices = vertices[edge]
            ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], edge_vertices[:, 2], 
                   color='r', lw=0.5)
        
        # Determine which face the arrow points to and draw diagonals
        arrow_direction = np.array([-np.sin(yaw), np.cos(yaw), 0])
        
        for face_idx, face in enumerate(faces):
            face_vertices = vertices[face]
            # Calculate face normal
            v1 = face_vertices[1] - face_vertices[0]
            v2 = face_vertices[2] - face_vertices[0]
            normal = np.cross(v1, v2)
            
            # If arrow points to this face
            if np.dot(normal, arrow_direction) < 0:
                # Draw diagonals on the front face
                diagonals = [
                    [face_vertices[0], face_vertices[2]],
                    [face_vertices[1], face_vertices[3]]
                ]
                for diag in diagonals:
                    ax.plot([diag[0][0], diag[1][0]], 
                           [diag[0][1], diag[1][1]], 
                           [diag[0][2], diag[1][2]], 
                           color='r', lw=0.5)
                break
    
    # Set view angle
    ax.view_init(elev=30, azim=290)
    
    # Set axis labels
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_zlabel('Z [m]', fontsize=12)
    
    # Set axis limits
    ax.set_xlim([-25, 25])
    ax.set_ylim([0, 50])
    ax.set_zlim([0, 50])
    
    # Set background transparent
    ax.set_facecolor('none')
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.grid(False)
    
    # Create save directory
    seq_name_no_ext = os.path.splitext(seq_name)[0]
    save_path = os.path.join(save_dir, seq_name_no_ext)
    os.makedirs(save_path, exist_ok=True)
    
    # Save figure
    plt.savefig(
        os.path.join(save_path, f'{name}.png'),
        bbox_inches='tight',
        pad_inches=0.1,
        dpi=300,
        transparent=True
    )
    plt.close()
    print(f"Plot saved successfully: {os.path.join(save_path, f'{name}.png')}")

def plot_label_fusion(lidar_set, name, seq_name, save_dir='./opencood/vis_training_v2xscenes/'):
    """
    Plot BEV visualization of fused LiDAR data from multiple vehicles with bounding boxes.
    
    Parameters
    ----------
    lidar_set : dict
        Dictionary containing:
        - 'lidar': list of point cloud arrays for each vehicle
        - 'cav_id': list of vehicle identifiers ('vehicle' or others)
        - 'bbx': list of bounding boxes (first element used for plotting)
    name : str
        Output filename
    seq_name : str
        Sequence name for directory organization
    save_dir : str
        Base directory for saving plots (default: './opencood/vis_training_v2xscenes/')
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    
    # Plot point clouds from all vehicles
    for i in range(len(lidar_set['lidar'])):
        # Different colors for ego vehicle vs others
        if lidar_set['cav_id'][i] == 'vehicle':
            ax.scatter(lidar_set['lidar'][i][:, 0], 
                      lidar_set['lidar'][i][:, 1], 
                      s=0.1, c='darkblue', marker='.')
        else:
            ax.scatter(lidar_set['lidar'][i][:, 0], 
                      lidar_set['lidar'][i][:, 1], 
                      s=0.1, c='gray', marker='.')
    
    # Plot bounding boxes (using first element of bbx list)
    for label in lidar_set['bbx'][0]:
        # Extract bounding box parameters
        x_center = label[0]
        y_center = label[1]
        z_center = label[2]
        height = label[3]   # h
        width = label[4]    # w
        length = label[5]   # l
        rotation = label[6] # yaw
        
        # Get 3D box vertices
        vertices = get_box_vertices(length, width, height, x_center, y_center, z_center, rotation)
        
        # Project to 2D (BEV)
        xy_vertices = vertices[:, :2]
        
        # Define edges for 2D projection
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
        
        # Draw each edge
        for edge in edges_2d:
            ax.plot(*zip(*edge), color='r', lw=1)
    
    # Set plot limits (LiDAR range)
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    
    # Set labels and ticks
    ax.set_xlabel('X [m]', fontsize=22)
    ax.set_ylabel('Y [m]', fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    
    # Create save directory
    seq_name_no_ext = os.path.splitext(seq_name)[0]
    save_path = os.path.join(save_dir, seq_name_no_ext)
    os.makedirs(save_path, exist_ok=True)
    
    # Save figure
    plt.savefig(
        os.path.join(save_path, f'{name}.png'),
        bbox_inches='tight',
        pad_inches=0.1,
        dpi=300
    )
    plt.close()
    print(f"Plot saved successfully: {os.path.join(save_path, f'{name}.png')}")