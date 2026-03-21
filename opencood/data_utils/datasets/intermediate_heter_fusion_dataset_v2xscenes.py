'''
-*- coding: utf-8 -*-
Author: Bowen Wang <98.wangbowen@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

intermediate heter fusion dataset

Note that for DAIR-V2X dataset,
Each agent should retrieve the objects itself, and merge them by iou, 
instead of using the cooperative label.
''' 

import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.utils.check_v2xscenes_plot import plot_label_bev, plot_label_fusion
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import merge_features_to_dict, compute_iou, convert_format
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json
from opencood.utils.heter_utils import Adaptor

PLOT = False

def getIntermediateheter_v2xscenesFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class IntermediateheterFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)

            # intermediate and supervise single
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']
            
            self.name = params['name']

            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.heterogeneous = True
            self.modality_assignment = None if ('assignment_path' not in params['heter'] or params['heter']['assignment_path'] is None) \
                                            else read_json(params['heter']['assignment_path'])
            
            self.ego_modality = params['heter']['ego_modality'] # "m1" or "m1&m2" or "m3"

            self.modality_name_list = list(params['heter']['modality_setting'].keys())
            self.sensor_type_dict = OrderedDict()

            lidar_channels_dict = params['heter'].get('lidar_channels_dict', OrderedDict())
            mapping_dict = params['heter']['mapping_dict']
            cav_preference = params['heter'].get("cav_preference", None)

            self.adaptor = Adaptor(self.ego_modality,
                                   self.modality_name_list,
                                   self.modality_assignment,
                                   lidar_channels_dict,
                                   mapping_dict,
                                   cav_preference,
                                   train)

            for modality_name, modal_setting in params['heter']['modality_setting'].items():
                self.sensor_type_dict[modality_name] = modal_setting['sensor_type']
                if modal_setting['sensor_type'] == 'lidar':
                    setattr(self, f"pre_processor_{modality_name}", build_preprocessor(modal_setting['preprocess'], train))

                elif modal_setting['sensor_type'] == 'camera':
                    setattr(self, f"data_aug_conf_{modality_name}", modal_setting['data_aug_conf'])

                else:
                    raise("Not support this type of sensor")

            self.kd_flag = params.get('kd_flag', False)

            self.box_align = False
            if "box_align" in params:
                self.box_align = True
                self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
                self.stage1_result = read_json(self.stage1_result_path)
                self.box_align_args = params['box_align']['args']
                


        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']
   
            transformation_matrix = np.array(selected_cav_base['params']['transformation_matrix'])
            transformation_matrix_clean = np.array(selected_cav_base['params']['transformation_matrix'])

            modality_name = selected_cav_base['modality_name']
            sensor_type = self.sensor_type_dict[modality_name]

            # lidar
            if sensor_type == "lidar" or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)

                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                                transformation_matrix)
                    
                lidar_np[:, :3] = projected_lidar
                selected_cav_processed.update({'projected_lidar': projected_lidar})
                
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:,:3] = projected_lidar

                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                    # 2023.8.31, to correct discretization errors. Just replace one point to avoid empty voxels. need fix later.
                    lidar_proj_np[np.random.randint(0, lidar_proj_np.shape[0]),:3] = np.array([0,0,0]) 
                    processed_lidar_proj = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_proj_np)
                    selected_cav_processed.update({f'processed_features_{modality_name}_proj': processed_lidar_proj})

                if sensor_type == "lidar":
                    processed_lidar = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_np)
                    selected_cav_processed.update({f'processed_features_{modality_name}': processed_lidar})

            # generate targets label single GT, note the reference pose is itself.
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
            )
            selected_cav_processed.update({
                                "single_label_dict": label_dict,
                                "single_object_bbx_center": object_bbx_center,
                                "single_object_bbx_mask": object_bbx_mask})

            # camera
            if sensor_type == "camera":
                camera_data_list = selected_cav_base["camera_data"]
                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        eval(f"self.data_aug_conf_{modality_name}"), self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)
                    

                selected_cav_processed.update(
                    {
                    f"image_inputs_{modality_name}": 
                        {
                            "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )

            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                        ego_pose_clean)

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )


            return selected_cav_processed

        def __getitem__(self, idx):
    
            base_data_dict = self.retrieve_base_data(idx)

            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
                
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            
            input_list_m1 = [] # can contain lidar or camera
            input_list_m2 = []
            input_list_m3 = []
            input_list_m4 = []

            agent_modality_list = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            exclude_agent = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            projected_lidar_stack = []
            projected_lidar_clean_list = [] # disconet

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []
                input_list_m1_proj = [] # 2023.8.31 to correct discretization errors with kd flag
                input_list_m2_proj = []
                input_list_m3_proj = []
                input_list_m4_proj = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)
                print(f"Distance from ego vehicle to RSU_{cav_id}: {distance} meters.")
                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    exclude_agent.append(cav_id)
                    continue
                
                # if modality not match
                if self.adaptor.unmatched_modality(selected_cav_base['modality_name']):
                    exclude_agent.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)   
                
            if len(cav_id_list) == 0:
                return None

            for cav_id in exclude_agent:
                base_data_dict.pop(cav_id)

            ########## Updated by Yifan Lu 2022.1.26 ############
            # box align to correct pose.
            # stage1_content contains all agent. Even out of comm range.
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content['cav_id_list'] # include those out of range
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent) for cur_agent in cur_agent_id_list] # indexing current agent in `all_agent_id_list`

                    pred_corners_list = [np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]

                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                        cur_agnet_pose, 
                                                                        uncertainty_list=uncertainty_list, 
                                                                        **self.box_align_args)
                        cur_agnet_pose[:,[0,1,4]] = refined_pose 

                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agnet_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agnet_pose[i].tolist()



            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
            
            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)
            lidar_set = {'lidar': [], 'cav_id': [], 'bbx': []}

            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                modality_name = selected_cav_base['modality_name']
                sensor_type = self.sensor_type_dict[selected_cav_base['modality_name']]

                # dynamic object center generator! for heterogeneous input
                if not self.visualize:
                    self.generate_object_center = eval(f"self.generate_object_center_{sensor_type}")
                # need discussion. In test phase, use lidar label.
                else: 
                    self.generate_object_center = self.generate_object_center_lidar

                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_cav_base)
                
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']


                if sensor_type == "lidar":
                    eval(f"input_list_{modality_name}").append(selected_cav_processed[f"processed_features_{modality_name}"])
                elif sensor_type == "camera":
                    eval(f"input_list_{modality_name}").append(selected_cav_processed[f"image_inputs_{modality_name}"])
                else:
                    raise
                
                agent_modality_list.append(modality_name)
                projected_lidar_stack.append(
                        selected_cav_processed['projected_lidar'])
                if self.visualize or self.kd_flag:
                    # heterogeneous setting do not support disconet' kd
                    projected_lidar_stack.append(
                        selected_cav_processed['projected_lidar'])
                    if sensor_type == "lidar" and self.kd_flag:
                        eval(f"input_list_{modality_name}_proj").append(selected_cav_processed[f"processed_features_{modality_name}_proj"])

                print(f'selected_cav_processed | cav_id: {cav_id}')
                bbx = selected_cav_processed['single_object_bbx_center'][0]
                print(f'  bbx_center: [{bbx[0]:.6f}, {bbx[1]:.6f}, {bbx[2]:.6f}, '
                    f'{bbx[3]:.6f}, {bbx[4]:.6f}, {bbx[5]:.6f}, {bbx[6]:.6f}]')
                
                if PLOT and cav_id == 'vehicle':
                    plot_label_bev(selected_cav_processed['projected_lidar'], selected_cav_processed['projected_lidar'], selected_cav_processed['object_bbx_center'], f'{self.name}_{idx}_check_{cav_id}_3D', self.name)      

                if self.supervise_single or self.heterogeneous:
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])
                
                lidar_set['lidar'].append(selected_cav_processed['projected_lidar'])
                lidar_set['cav_id'].append(cav_id)
                # lidar_set['bbx'].append(selected_cav_processed['object_bbx_center'])

            # generate single view GT label
            if self.supervise_single or self.heterogeneous:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                    })
            
            # exculude all repetitve objects, DAIR-V2X
            if self.params['fusion']['dataset'] == 'dairv2x' or 'massv2x' or "v2xscenes":
                if len(object_stack) == 1:
                    object_stack = object_stack[0]
                else:
                    ego_boxes_np = object_stack[0]
                    cav_boxes_np = object_stack[1]
                    order = self.params['postprocess']['order']
                    ego_corners_np = box_utils.boxes_to_corners_3d(ego_boxes_np, order)
                    cav_corners_np = box_utils.boxes_to_corners_3d(cav_boxes_np, order)
                    ego_polygon_list = list(convert_format(ego_corners_np))
                    cav_polygon_list = list(convert_format(cav_corners_np))
                    iou_thresh = 0.05 


                    gt_boxes_from_cav = []
                    for i in range(len(cav_polygon_list)):
                        cav_polygon = cav_polygon_list[i]
                        ious = compute_iou(cav_polygon, ego_polygon_list)
                        if (ious > iou_thresh).any():
                            continue
                        gt_boxes_from_cav.append(cav_boxes_np[i])
                    
                    if len(gt_boxes_from_cav):
                        object_stack_from_cav = np.stack(gt_boxes_from_cav)
                        object_stack = np.vstack([ego_boxes_np, object_stack_from_cav])
                    else:
                        object_stack = ego_boxes_np

                unique_indices = np.arange(object_stack.shape[0])
                object_id_stack = np.arange(object_stack.shape[0])
            else:
                # exclude all repetitive objects, OPV2V-H
                unique_indices = \
                    [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            
            for modality_name in self.modality_name_list:
                if self.sensor_type_dict[modality_name] == "lidar":
                    merged_feature_dict = merge_features_to_dict(eval(f"input_list_{modality_name}")) 
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_feature_dict}) # maybe None
                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(eval(f"input_list_{modality_name}"), merge='stack')
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_image_inputs_dict}) # maybe None

            if self.kd_flag:
                # heterogenous setting do not support DiscoNet's kd
                # stack_lidar_np = np.vstack(projected_lidar_stack)
                # stack_lidar_np = mask_points_by_range(stack_lidar_np,
                #                             self.params['preprocess'][
                #                                 'cav_lidar_range'])
                # stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                for modality_name in self.modality_name_list:
                    processed_data_dict['ego'].update({
                        f'input_{modality_name}_proj': merge_features_to_dict(eval(f"input_list_{modality_name}_proj")) # maybe None
                        })


            processed_data_dict['ego'].update({'agent_modality_list': agent_modality_list})

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=self.anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses})
            
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})

            processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})
            
            lidar_set['bbx'].append(object_bbx_center)
            if PLOT:
                plot_label_fusion(lidar_set, f'{self.name}_{idx}_check_lidar_fusion', self.name) 
            # plot_label_bev(processed_data_dict['ego']['origin_lidar'], processed_data_dict['ego']['origin_lidar'], object_bbx_center, f'{self.name}_check_fusion') 
            return processed_data_dict


        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            inputs_list_m1 = [] 
            inputs_list_m2 = []
            inputs_list_m3 = []
            inputs_list_m4 = []

            inputs_list_m1_proj = [] 
            inputs_list_m2_proj = []
            inputs_list_m3_proj = []
            inputs_list_m4_proj = []

            agent_modality_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            lidar_pose_clean_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single or self.heterogeneous:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

                for modality_name in self.modality_name_list:
                    if ego_dict[f'input_{modality_name}'] is not None:
                        eval(f"inputs_list_{modality_name}").append(ego_dict[f'input_{modality_name}']) # OrderedDict() if empty?

                agent_modality_list.extend(ego_dict['agent_modality_list'])
                
                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                origin_lidar.append(ego_dict['origin_lidar'])

                if self.kd_flag:
                    # hetero setting do not support disconet' kd
                    # teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])
                    for modality_name in self.modality_name_list:
                        if ego_dict[f'input_{modality_name}_proj'] is not None:
                            eval(f"inputs_list_{modality_name}_proj").append(ego_dict[f"input_{modality_name}_proj"])

                ### 2022.10.10 single gt ####
                if self.supervise_single or self.heterogeneous:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])


            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))


            # 2023.2.5
            for modality_name in self.modality_name_list:
                if len(eval(f"inputs_list_{modality_name}")) != 0:
                    if self.sensor_type_dict[modality_name] == "lidar":
                        merged_feature_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"))
                        processed_lidar_torch_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(merged_feature_dict)
                        output_dict['ego'].update({f'inputs_{modality_name}': processed_lidar_torch_dict})

                    elif self.sensor_type_dict[modality_name] == "camera":
                        merged_image_inputs_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"), merge='cat')
                        output_dict['ego'].update({f'inputs_{modality_name}': merged_image_inputs_dict})


            output_dict['ego'].update({"agent_modality_list": agent_modality_list})
            
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                     'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len
            

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'record_len': record_len,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'lidar_pose_clean': lidar_pose_clean,
                                    'lidar_pose': lidar_pose,
                                    'anchor_box': self.anchor_box_torch})


  
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.kd_flag:
                # teacher_processed_lidar_torch_dict = \
                #     self.pre_processor.collate_batch(teacher_processed_lidar_list)
                # output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})
                for modality_name in self.modality_name_list:
                    if len(eval(f"inputs_list_{modality_name}_proj")) != 0 and self.sensor_type_dict[modality_name] == "lidar":
                        merged_feature_proj_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}_proj"))
                        processed_lidar_torch_proj_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(merged_feature_proj_dict)
                        output_dict['ego'].update({f'inputs_{modality_name}_proj': processed_lidar_torch_proj_dict})

            if self.supervise_single  or self.heterogeneous:
                output_dict['ego'].update({
                    "label_dict_single":{
                            "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                            "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                            "targets": torch.cat(targets_single, dim=0),
                            # for centerpoint
                            "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                            "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                        },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                })

            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            if batch[0] is None:
                return None
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                    self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch,
                                        'transformation_matrix_clean':
                                        transformation_matrix_clean_torch,})

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list'],
                "agent_modality_list": batch[0]['ego']['agent_modality_list']
            })

            return output_dict


        def post_process(self, data_dict, output_dict, save_trk=False):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)  #TODO


            if save_trk:
                gt_box_tensor, gt_object_id_tensor = self.post_processor.generate_gt_bbx(data_dict, save_trk)  # TODO 这里要加gt_object_id_tensor

                return pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor
            else:
                gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)  # TODO 这里要加gt_object_id_tensor

                return pred_box_tensor, pred_score, gt_box_tensor
        
        
        def pose_to_tfm(pose: np.ndarray) -> np.ndarray:
            """
            Convert [x, y, z, roll, yaw, pitch] to a 4x4 transformation matrix.
            The input pose is in the format [x, y, z, roll, yaw, pitch] where:
                - x, y, z are the position coordinates
                - roll, yaw, pitch are the rotation angles in radians
            """
            x, y, z, roll, yaw, pitch = pose

            # Calculate rotation matrix from Euler angles (roll, yaw, pitch)
            
            # Rotation around X-axis (roll)
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])

            # Rotation around Y-axis (pitch)
            R_y = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])

            # Rotation around Z-axis (yaw)
            R_z = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])

            # Combined rotation matrix (XYZ order: roll-pitch-yaw)
            R = np.dot(R_z, np.dot(R_y, R_x))

            # Construct the full transformation matrix (4x4)
            tfm = np.eye(4)
            tfm[:3, :3] = R
            tfm[:3, 3] = [x, y, z]

            return tfm

    return IntermediateheterFusionDataset

if __name__ == "__main__":
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets.basedataset.v2xscenes_basedataset import V2XSCENESBaseDataset
    params_path = "/home/myData/storage/code/HEAL/opencood/hypes_yaml/massv2x/v2xsences_cobevt.yaml"
    dataset_cfg = yaml_utils.load_yaml(params_path)
    IntermediateFusionDataset = getIntermediateheter_v2xscenesFusionDataset(V2XSCENESBaseDataset)
    intermediate_fusion_dataset = IntermediateFusionDataset(dataset_cfg, visualize=False)
