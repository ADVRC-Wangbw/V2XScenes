import os
from collections import OrderedDict
from idlelib.pyparse import trans

import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pickle
import json
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import x1_to_x2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from scipy.spatial.transform import Rotation as R

class V2XSCENESBaseDataset(Dataset):
    def __init__(self, params, visualize=False, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.use_hdf5 = True
        self.name = params['name']        
        self.label_num = set()

        # Pre- and post-processing
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

        # Data augmentation (early/late vs intermediate)
        self.data_augmentor = (
            DataAugmentor(params['data_augment'], train)
            if 'data_augment' in params else None
        )

        # Root directory based on train/test mode
        self.root_dir = params['root_dir'] if self.train else params['test_dir']
        print("Dataset dir:", self.root_dir)

        # Latency setting
        self.latency = params.get('latency', 0.0)
        if 'latency' in params:
            print("Latency in V2xScenesDataset:", params['latency'])

        # Road calibration
        self.road_calib = None
        if 'road_calib_path' in params:
            road_calib_path = params['road_calib_path']
            if os.path.exists(road_calib_path):
                with open(road_calib_path, "r") as f:
                    self.road_calib = json.load(f)

        self.road_calib_shift = np.array(params['road_calib_shift']) if 'road_calib_shift' in params else None

        # Vehicle pose (single file)
        self.veh_pose = None
        if 'veh_pose_path' in params:
            with open(params['veh_pose_path'], 'rb') as f:
                self.veh_pose = pickle.load(f)

        # Vehicle pose list (multiple scenarios)
        self.veh_pose_list = []
        if 'veh_json_path' in params:
            scenario_list = params['train_scenario_list'] if self.train else params['test_scenario_list']
            for scenario in scenario_list:
                json_path = os.path.join(params['veh_json_path'], f"icp_results_{scenario}.json")
                with open(json_path, 'r') as f:
                    self.veh_pose_list.append(json.load(f))

        # CAV list
        self.cav_list = params.get("cav_list", [])

        # Max number of CAVs
        self.max_cav = params.get('train_params', {}).get('max_cav', 5)

        # Input source flags
        input_source = params.get('input_source', [])
        self.load_lidar_file = 'lidar' in input_source or self.visualize
        self.load_camera_file = 'camera' in input_source
        self.load_depth_file = 'depth' in input_source

        # Label type and center generation
        self.label_type = params['label_type']  # 'lidar' or 'camera'
        if self.label_type == "lidar":
            self.generate_object_center = self.generate_object_center_lidar
        else:
            self.generate_object_center = self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center  # follows the above

        # Camera data augmentation config
        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # Additional data extensions
        self.add_data_extension = params.get('add_data_extension', [])

        # Noise setting
        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict(add_noise=False)

        # Scenario folders
        if self.train and "train_scenario_list" in params:
            self.scenario_folders = [os.path.join(self.root_dir, x) for x in params["train_scenario_list"]]
        elif not self.train and "test_scenario_list" in params:
            self.scenario_folders = [os.path.join(self.root_dir, x) for x in params["test_scenario_list"]]
        else:
            self.scenario_folders = [0]
            print("No scenario list found; using default [0]")

        # Train mode
        self.train_mode = params.get("train_mode")

        # Final reinitialization
        self.reinitializev2x()

    def get_element_by_idx(self, data, idx):
        keys = list(data.keys())
        if idx < 0 or idx >= len(keys):
            raise IndexError("Index out of range")
        key = keys[idx]
        return data[key]

    def tfm_to_pose(self, tfm: np.ndarray):
        """
        turn transformation matrix to [x, y, z, roll, yaw, pitch]
        we use radians format.
        tfm is pose in transformation format, and XYZ order, i.e. roll-pitch-yaw
        """
        # There forumlas are designed from x_to_world, but equal to the one below.
        yaw = np.degrees(np.arctan2(tfm[1, 0], tfm[0, 0]))  # clockwise in carla

        roll = np.degrees(np.arctan2(-tfm[2, 1], tfm[2, 2]))  # but counter-clockwise in carla
        pitch = np.degrees(
            np.arctan2(tfm[2, 0], ((tfm[2, 1] ** 2 + tfm[2, 2] ** 2) ** 0.5)))  # but counter-clockwise in carla

        x, y, z = tfm[:3, 3]
        return ([x, y, z, roll, yaw, pitch])

    def find_timestamp_with_exact_difference(self, data, target_timestamp, target_difference=0.1):
        target = float(target_timestamp)

        for key in data.keys():
            if abs(float(key) - target) < target_difference:
                return key, data[key]

    def reinitializev2x(self):
        self.scenario_database = OrderedDict()
        self.len_record = []
        self.scenario_database['name'] = self.name
        cav_list = self.cav_list

        for (j, scenario_folder) in enumerate(self.scenario_folders):  # TODO
            
            print(f"Loaded scenario:{scenario_folder}")
            if self.veh_pose_list is not None:
                veh_pose_json = self.veh_pose_list[j]
            else:
                veh_pose_json = self.veh_pose_json
            self.scenario_database[j] = OrderedDict()
            for (i, cav_id) in enumerate(cav_list):
                self.scenario_database[j][cav_id] = OrderedDict()

                tmp_list = []
                seq_length = len(sorted([os.path.join(os.path.join(scenario_folder, "label_new", "vehicle_global_label"), x)
                                for x in os.listdir(os.path.join(scenario_folder, "label_new", "vehicle_global_label")) if
                                x.endswith('.json') and 'additional' not in x]))
                # print("cooperative_seq_length", seq_length)
                
                if cav_id.startswith("vehicle"):
                    label_path = os.path.join(scenario_folder, "label_new", "vehicle_global_label")  #todo 新修改albelnew
                    pcd_path = os.path.join(scenario_folder, "veh_lidar", "middle")  # TODO
                    json_files = \
                        sorted([os.path.join(label_path, x)
                                for x in os.listdir(label_path) if
                                x.endswith('.json') and 'additional' not in x])

                    pcd_files = \
                        sorted([os.path.join(pcd_path, x)
                                for x in os.listdir(pcd_path) if
                                x.endswith('.pcd') and 'additional' not in x])
                        
                    if '21' in cav_list or '27' in cav_list:
                        label_path_21 = os.path.join(scenario_folder, "label_new", "sort_road_lidar_label", '21')  # todo  修改 label_new
                        label_path_27 = os.path.join(scenario_folder, "label_new", "sort_road_lidar_label", '27')  # todo  修改 label_new
                        
                        json_files_21 = \
                            sorted([os.path.join(label_path_21, x)
                                    for x in os.listdir(label_path_21) if
                                    x.endswith('.json') and 'additional' not in x])
                            
                        json_files_27 = \
                            sorted([os.path.join(label_path_27, x)
                                    for x in os.listdir(label_path_27) if
                                    x.endswith('.json') and 'additional' not in x])
                            
                        print("seq_length, len(json_files), len(json_files_21), len(json_files_27)", seq_length, len(json_files), len(json_files_21), len(json_files_27))
                        
                        if len(json_files) != len(json_files_21) or len(json_files) != len(json_files_27):
                            if len(json_files_21) < len(json_files_27):
                                timestamps_idx = [timestamp for timestamp in range(len(json_files_21))]
                                timestamps = [os.path.splitext(os.path.basename(file))[0] for file in json_files_21]
                            else:
                                timestamps_idx = [timestamp for timestamp in range(len(json_files_27))]
                                timestamps = [os.path.splitext(os.path.basename(file))[0] for file in json_files_27]
                        else:
                            timestamps_idx = [timestamp for timestamp in range(len(json_files))]
                            timestamps = [os.path.splitext(os.path.basename(file))[0] for file in json_files]
                    else:
                        timestamps_idx = [timestamp for timestamp in range(len(json_files))]
                        timestamps = [os.path.splitext(os.path.basename(file))[0] for file in json_files]   
                    print("seq_length, len(timestamps_idx)", seq_length, len(timestamps_idx))
                    
                    for timestamp in timestamps_idx:
                        self.scenario_database[j][cav_id][timestamp] = OrderedDict()

                        lidar_file = pcd_files[timestamp]
                        json_file = json_files[timestamp]
                        self.scenario_database[j][cav_id][timestamp]['lidar'] = lidar_file
                        self.scenario_database[j][cav_id][timestamp]['json'] = json_file
                        self.scenario_database[j][cav_id][timestamp]['timestamp'] = timestamps[timestamp]

                        self.scenario_database[j][cav_id][timestamp]['calib'] = OrderedDict()

                        self.scenario_database[j][cav_id][timestamp]['calib'][
                            'transformation_matrix'] = np.eye(4)

                        self.scenario_database[j][cav_id][timestamp]['calib']['lidar_pose'] = self.tfm_to_pose(
                            np.eye(4))

                    self.scenario_database[j][cav_id]['ego'] = True
                    tmp_list.append(len(timestamps))

                else:
                    label_path = os.path.join(scenario_folder, "label_new", "sort_road_lidar_label", cav_id)  # todo  修改 label_new
                    pcd_path = os.path.join(scenario_folder, "road_lidar", f"msop_{cav_id}")
                    
                    json_files = \
                        sorted([os.path.join(label_path, x)
                                for x in os.listdir(label_path) if
                                x.endswith('.json') and 'additional' not in x])
                    
                    pcd_files = \
                        sorted([os.path.join(pcd_path, x)
                                for x in os.listdir(pcd_path) if
                                x.endswith('.pcd') and 'additional' not in x])
                        
                    if '21' in cav_list or '27' in cav_list:
                        label_path_21 = os.path.join(scenario_folder, "label_new", "sort_road_lidar_label", '21')  # todo  修改 label_new
                        label_path_27 = os.path.join(scenario_folder, "label_new", "sort_road_lidar_label", '27')  # todo  修改 label_new
                        
                        json_files_21 = \
                            sorted([os.path.join(label_path_21, x)
                                    for x in os.listdir(label_path_21) if
                                    x.endswith('.json') and 'additional' not in x])
                            
                        json_files_27 = \
                            sorted([os.path.join(label_path_27, x)
                                    for x in os.listdir(label_path_27) if
                                    x.endswith('.json') and 'additional' not in x])
                            
                        print(f"seq_length_{cav_id}, len(json_files), len(json_files_21), len(json_files_27)", seq_length, len(json_files), len(json_files_21), len(json_files_27))
                        
                        if len(json_files) != len(json_files_21) or len(json_files) != len(json_files_27):
                            if len(json_files_21) < len(json_files_27):
                                timestamps_idx = [timestamp for timestamp in range(len(json_files_21))]
                                timestamps = [os.path.splitext(os.path.basename(file))[0] for file in json_files_21]
                            else:
                                timestamps_idx = [timestamp for timestamp in range(len(json_files_27))]
                                timestamps = [os.path.splitext(os.path.basename(file))[0] for file in json_files_27]
                        else:
                            timestamps_idx = [timestamp for timestamp in range(len(json_files))]
                            timestamps = [os.path.splitext(os.path.basename(file))[0] for file in json_files]
                    else:
                        timestamps_idx = [timestamp for timestamp in range(len(json_files))]
                        timestamps = [os.path.splitext(os.path.basename(file))[0] for file in json_files]  
                
                    for timestamp in timestamps_idx:
                        
                        # Calculate the adjusted timestamp with latency
                        adjusted_timestamp = float(timestamps[timestamp]) - float(self.latency)
                        closest_timestamp = None
                        for t in timestamps_idx:
                            if abs(float(timestamps[t]) - adjusted_timestamp) <= 0.1:  # Match within ±0.1s
                                closest_timestamp = t
                                break

                        if closest_timestamp is not None:
                            closest_timestamp = closest_timestamp
                        else:
                            closest_timestamp = 0
                            
                        # print("closest_timestamp", closest_timestamp, "timestamps", timestamp)
                        self.scenario_database[j][cav_id][timestamp] = OrderedDict()
                        self.scenario_database[j][cav_id][timestamp]['lidar'] = pcd_files[closest_timestamp]
                        self.scenario_database[j][cav_id][timestamp]['json'] = json_files[closest_timestamp]
                        self.scenario_database[j][cav_id][timestamp]['timestamp'] = closest_timestamp

                        self.scenario_database[j][cav_id][timestamp]['calib'] = OrderedDict()
                        key, value = self.find_timestamp_with_exact_difference(veh_pose_json, timestamps[closest_timestamp])
                        
                        transformation_matrix_16 = value["final_transformation"]
                        vehicle_pose = np.array(transformation_matrix_16).reshape(4, 4)
                        
                        transformation_4x4 = np.array([
                            -0.6511579155921936, -0.7583672404289246, -0.02952973172068596, 69.85047912597656,
                            0.7589206695556641, -0.6503510475158691, -0.03289959207177162, 154.08926391601563,
                            0.0057452027685940266, -0.043833568692207336, 0.9990233182907104, 5.200427532196045,
                            0.0, 0.0, 0.0, 1.0
                        ]).reshape(4, 4)
 
                        T_vehlidar_to_world = np.dot(np.linalg.inv(np.array(transformation_4x4).reshape(4, 4)),
                                                     np.array(vehicle_pose).reshape(4, 4))

                        roadlidar_to_world = self.road_calib[str(cav_id)] 

                        roadlidar_to_world = np.array(roadlidar_to_world)
                       
                        if self.train_mode == 'cooperative':
                            self.scenario_database[j][cav_id][timestamp]['calib'][
                                'transformation_matrix'] = np.dot(np.linalg.inv(np.array(T_vehlidar_to_world)), roadlidar_to_world)
                            self.scenario_database[j][cav_id][timestamp]['calib']['lidar_pose'] = self.tfm_to_pose(
                            np.dot(np.linalg.inv(np.array(T_vehlidar_to_world)), roadlidar_to_world))
                            self.scenario_database[j][cav_id][timestamp]['calib']['vehicle_to_global'] = T_vehlidar_to_world
                        else:
                            self.scenario_database[j][cav_id][timestamp]['calib'][
                                'transformation_matrix'] = roadlidar_to_world
                            self.scenario_database[j][cav_id][timestamp]['calib']['lidar_pose'] = self.tfm_to_pose(roadlidar_to_world)
                            self.scenario_database[j][cav_id][timestamp]['calib']['vehicle_to_global'] = np.eye(4) 
                            
                    self.scenario_database[j][cav_id]['ego'] = True
                    tmp_list.append(len(timestamps))

            min_tmp = min(tmp_list)
            if not self.len_record:
                self.len_record.append(min_tmp)
            else:
                prev_last = self.len_record[-1]
                self.len_record.append(prev_last + min_tmp)
            print('self.len_record', self.len_record)

    def process_rotation(self, angle):
        r = R.from_euler('z', angle)
        euler = r.as_euler('xyz', degrees=True)
        return euler

    def transform_position_and_orientation(self, x_center, y_center, z_center, z_rotation, T):

        original_position = np.array([x_center, y_center, z_center, 1.0])
        combined_transform = T
        transformed_position = np.dot(combined_transform, original_position)
        transformed_x_center, transformed_y_center, transformed_z_center, _ = transformed_position
        delta_z_rotation = np.arctan2(combined_transform[1, 0], combined_transform[0, 0])
        new_z_rotation = z_rotation + delta_z_rotation  

        return transformed_x_center, transformed_y_center, transformed_z_center, new_z_rotation

    def process_params_v2xscenes_rsu(self, params, vehicle_to_world):
        data = OrderedDict()
        data['vehicles'] = OrderedDict()
        if "class" in self.params["postprocess"]:
            detection_class = self.params["postprocess"]["class"]
        else:
            detection_class = "Car"
        for cav in params:
            if (cav["type"] == "Truck" and cav["type"] in detection_class and cav["3d_dimensions"]['l'] > 9)  or (cav["type"] != "Truck" and cav["type"] in detection_class):
                id = cav["track_id"]
                loc = cav["3d_location"]  # x, y, z
                dim = cav["3d_dimensions"]  # h, l, w

                bbx = OrderedDict()

                transformed_x_center, transformed_y_center, transformed_z_center, new_z_rotation \
                    = self.transform_position_and_orientation(loc['x'], loc['y'], loc['z'], cav['rotation'],
                                                              np.linalg.inv(vehicle_to_world))

                extent_for_bbx = [dim['h'] / 2, dim['w'] / 2, dim['l'] / 2]
                loc_veh = [transformed_x_center, transformed_y_center, transformed_z_center]

                bbx['angle'] = self.process_rotation(new_z_rotation)
                bbx['center'] = [0, 0, 0]
                bbx['location'] = loc_veh
                bbx['extent'] = extent_for_bbx
                data['vehicles'][id] = bbx
        return data

    def transform_point(self, point, transform_matrix):  

        homogeneous_point = np.array([point[0], point[1], point[2], 1])
        transformed_point = np.dot(transform_matrix, homogeneous_point)
        return transformed_point[:3]

    def process_params_v2xscenes_veh(self, params,
                                   lidar_2_global):  # TODO 处理车端params 区别是每个bbx里面loc_veh要和4x4的transformation_matrix相乘
        data = OrderedDict()
        data['vehicles'] = OrderedDict()
        if "class" in self.params["postprocess"]:
            detection_class = self.params["postprocess"]["class"]
        else:
            detection_class = "Car"
        for cav in params:
            self.label_num.add(cav["type"])
            if (cav["type"] == "Truck" and cav["type"] in detection_class and cav["3d_dimensions"]['l'] > 9)  or (cav["type"] != "Truck" and cav["type"] in detection_class):  #todo
                id = cav["track_id"]
                loc = cav["3d_location"]  # x, y, z
                dim = cav["3d_dimensions"]  # h, l, w

                veh_2_ground = np.array([[0.999657, -0.00109618, 0.026154, 0],
                                         [0, 0.999123, 0.0418757, 0],
                                         [-0.0261769, -0.0418613, 0.99878, 0],
                                         [0, 0, 0, 1]])

                veh_2_ground = np.eye(4)

                T1_inv = np.linalg.inv(veh_2_ground)

                combined_transform = np.dot(lidar_2_global, T1_inv)

                combined_transform = np.eye(4)

                transformed_x_center, transformed_y_center, transformed_z_center, new_z_rotation \
                    = self.transform_position_and_orientation(loc['x'], loc['y'], loc['z'], cav['rotation'],
                                                              combined_transform)  # 标签直接转到世界坐标系

                extent_for_bbx = [dim['h'] / 2, dim['w'] / 2, dim['l'] / 2]
                loc_veh = [transformed_x_center, transformed_y_center, transformed_z_center]

                bbx = OrderedDict()
                bbx['angle'] = self.process_rotation(new_z_rotation)
                bbx['center'] = [0, 0, 0]
                bbx['location'] = loc_veh
                bbx['extent'] = extent_for_bbx
                data['vehicles'][id] = bbx

        return data

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0

        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key

        # print(scenario_database.keys())
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        
        data = OrderedDict()
        # data['name'] = self.scenario_database['name']
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            
            # todo load param file: json is faster than yaml
            # print(cav_id, timestamp_key, len(cav_content))
            # print(cav_content[timestamp_key].keys())
            if 'yaml' in cav_content[timestamp_key]:
                json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            else:
                json_file = cav_content[timestamp_key]['json']

            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    params = json.load(f)
                    if cav_id.startswith("vehicle"):
                        lidar_2_global = scenario_database[cav_id][timestamp_key]['calib']['transformation_matrix']
                        params = self.process_params_v2xscenes_veh(params, lidar_2_global)
                    else:
                        vehicle_2_global = scenario_database[cav_id][timestamp_key]['calib']['vehicle_to_global']
                        params = self.process_params_v2xscenes_rsu(params, vehicle_2_global)

                    data[cav_id]['params'] = params
            else:
                data[cav_id]['params'] = \
                    load_yaml(cav_content[timestamp_key]['yaml'])

            lidar_pose = cav_content[timestamp_key]['calib']['lidar_pose']
            lidar_2_world_4x4 = cav_content[timestamp_key]['calib']['transformation_matrix']
            data[cav_id]['params']['transformation_matrix'] = lidar_2_world_4x4
            data[cav_id]['params']['lidar_pose'] = lidar_pose
            data[cav_id]['params']['lidar_pose_clean'] = lidar_pose
            # todo
            # load camera file: hdf5 is faster than png
            if 'cameras' in cav_content[timestamp_key]:

                hdf5_file = cav_content[timestamp_key]['cameras'][0].replace("camera0.png", "imgs.hdf5")

                if self.use_hdf5 and os.path.exists(hdf5_file):
                    with h5py.File(hdf5_file, "r") as f:
                        data[cav_id]['camera_data'] = []
                        data[cav_id]['depth_data'] = []
                        for i in range(4):
                            if self.load_camera_file:
                                data[cav_id]['camera_data'].append(Image.fromarray(f[f'camera{i}'][()]))
                            if self.load_depth_file:
                                data[cav_id]['depth_data'].append(Image.fromarray(f[f'depth{i}'][()]))
                else:
                    if self.load_camera_file:
                        data[cav_id]['camera_data'] = \
                            load_camera_data(cav_content[timestamp_key]['cameras'])
                    if self.load_depth_file:
                        data[cav_id]['depth_data'] = \
                            load_camera_data(cav_content[timestamp_key]['depths'])

            # load lidar file
            if self.load_lidar_file or self.visualize:
                data[cav_id]['lidar_np'] = \
                    pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

            if getattr(self, "heterogeneous", False):
                # data[cav_id]['modality_name'] = cav_content[timestamp_key]['modality_name']
                data[cav_id]['modality_name'] = "m1"

            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                if not os.path.exists(cav_content[timestamp_key][file_extension]):
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace(
                        "train", "additional/train")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace(
                        "validate", "additional/validate")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace(
                        "test", "additional/test")

                if '.yaml' in file_extension:
                    data[cav_id][file_extension] = \
                        load_yaml(cav_content[timestamp_key][file_extension])
                else:
                    data[cav_id][file_extension] = \
                        cv2.imread(cav_content[timestamp_key][file_extension])
        return data

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        sensor : str
            "camera" or "depth"

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def generate_object_center_lidar(self,
                                     cav_contents,
                                     reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center_v2xscenes(cav_contents,
                                                          reference_lidar_pose)

    def generate_object_center_camera(self,
                                      cav_contents,
                                      reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose
        )

    def get_ext_int(self, params, camera_id):
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(
            np.float32)
        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
            np.float32
        )
        return camera_to_lidar, camera_intrinsic

    def reinitialize(self):
        
        # cav_list = ["msop_6691, msop_6692"]
        
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            if self.train:
                cav_list = [x for x in os.listdir(scenario_folder)
                            if os.path.isdir(
                        os.path.join(scenario_folder, x))]
                # cav_list = sorted(cav_list)
                random.shuffle(cav_list)
            else:
                cav_list = sorted([x for x in os.listdir(scenario_folder)
                                   if os.path.isdir(
                        os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            """
            roadside unit data's id is always negative, so here we want to
            make sure they will be in the end of the list as they shouldn't
            be ego vehicle.
            """
            print(cav_list)
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            """
            make the first cav to be ego modality
            """
            if getattr(self, "heterogeneous", False):
                scenario_name = scenario_folder.split("/")[-1]
                cav_list = self.adaptor.reorder_cav_list(cav_list, scenario_name)

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])

                # this timestamp is not ready
                yaml_files = [x for x in yaml_files if not ("2021_08_20_21_10_24" in x and "000265" in x)]

                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.find_camera_files(cav_path,
                                                          timestamp)
                    depth_files = self.find_camera_files(cav_path,
                                                         timestamp, sensor="depth")
                    depth_files = [depth_file.replace("OPV2V", "OPV2V_Hetero") for depth_file in depth_files]

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp]['depths'] = \
                        depth_files

                    if getattr(self, "heterogeneous", False):
                        scenario_name = scenario_folder.split("/")[-1]
                        cav_modality = self.adaptor.reassign_cav_modality(
                            self.modality_assignment[scenario_name][cav_id], j)

                        self.scenario_database[i][cav_id][timestamp]['modality_name'] = cav_modality

                        self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                            self.adaptor.switch_lidar_channels(cav_modality, lidar_file)

                    # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False
