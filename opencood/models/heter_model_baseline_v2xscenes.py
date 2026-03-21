# -*- coding: utf-8 -*-
# Author: Bowen Wang <98.wangbowen@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A unified framework for LiDAR-only / Camera-only / Heterogeneous collaboration.
# Support multiple fusion strategies.


import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict, Counter
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, CoBEVT, Where2commFusion, Who2comFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision
import sys

class HeterModelBaselineV2xscenes(nn.Module):
    def __init__(self, args):
        super(HeterModelBaselineV2xscenes, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]    #todo 这应该三个m1吧
        self.modality_name_list = modality_name_list

        self.ego_modality = args['ego_modality']

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(model_setting['backbone_args'], 
                                                                       model_setting['backbone_args'].get('inplanes',64)))

            """
            shrink conv building
            """
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'], kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7, kernel_size=1))
            setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))


        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        if args['fusion_method'] == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'cobevt':
            self.fusion_net = CoBEVT(args['cobevt'])
        if args['fusion_method'] == 'where2comm':
            self.fusion_net = Where2commFusion(args['where2comm']['feat_dim'])
        if args['fusion_method'] == 'who2com':
            self.fusion_net = Who2comFusion(args['who2com'])


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])
            self.model_train_init()


        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        
        print("modality_feature_dict shapes:")
        for modality_name in modality_feature_dict.keys():
            print(f"{modality_name}: {modality_feature_dict[modality_name].shape}")

        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)
        
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        """
        Single supervision
        """
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})

        """
        Feature Fusion (multiscale).

        we omit self.backbone's first layer.
        """
        print('record_len', record_len)
        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)
        print(f"Fused feature shape: {fused_feature.shape}")

        import matplotlib.pyplot as plt
        import os
        from itertools import count
        import time
        import shutil
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        def visualize_and_save_bev_feature_auto(
            fused_feature: torch.Tensor,
            mode: str = 'mean',
            channel_idx: int = 0,
            title: str = "BEV Feature Heatmap",
            save_dir: str = 'output/heatmaps',
            show: bool = False,
            prefix: str = 'bev_feature'
        ):
            """
            Visualize and save BEV feature map with auto-increment filename.

            Args:
                fused_feature (torch.Tensor): Feature tensor of shape (1, C, H, W)
                mode (str): 'mean' or 'channel'
                channel_idx (int): Channel index if mode == 'channel'
                title (str): Plot title
                save_dir (str): Directory to save heatmaps
                show (bool): Whether to display plot
                prefix (str): Prefix for saved file name
            """
            # Validate input
            assert fused_feature.dim() == 4 and fused_feature.size(0) == 1
            
            # Create save directory
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate heatmap based on mode
            if mode == 'mean':
                heatmap = fused_feature.mean(dim=1).squeeze().cpu().numpy()
                suffix = 'mean'
            elif mode == 'channel':
                assert 0 <= channel_idx < fused_feature.size(1)
                heatmap = fused_feature[0, channel_idx].cpu().numpy()
                suffix = f'channel{channel_idx}'
            else:
                raise ValueError("Invalid mode: choose 'mean' or 'channel'")
            
            # Generate filename with timestamp
            timestamp = int(time.time())
            save_path = os.path.join(save_dir, f"{prefix}_{suffix}_{timestamp:03d}.png")
            
            # Create plot
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Draw heatmap
            im = ax.imshow(heatmap, cmap='Spectral_r', origin='lower', vmin=0, vmax=0.1)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=12)
            
            # Set axis limits and ticks
            Xmin, Xmax = 35, 125
            Ymin, Ymax = 35, 125
            ax.set_xlim([Xmin, Xmax])
            ax.set_ylim([Ymin, Ymax])
            ax.set_xticks(range(Xmin, Xmax + 1, 20))
            ax.set_yticks(range(Ymin, Ymax + 1, 20))
            ax.tick_params(axis='both', labelsize=12)
            
            # Save figure
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"[{timestamp}] Saved heatmap to: {save_path}")
            
            # Show or close
            if show:
                plt.show()
            else:
                plt.close()
        
        # visualize_and_save_bev_feature_auto(
        #     fused_feature,
        #     mode='mean',
        #     save_dir='/home/myData/storage/code/HEAL/opencood/models/plot2/cobevt/',
        #     prefix='mybev',
        #     show=False
        # )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})

        return output_dict
