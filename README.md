# V2XScenes

**[ICCV 2025]** V2XScenes: A Multiple Challenging Traffic Conditions Dataset for Large-Range Vehicle-Infrastructure Collaborative Perception

This repository contains a real-world **multi-challenging-condition dataset** covering large-range road sections for **multi-modal V2X cooperative perception** research.

**[Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_V2XScenes_A_Multiple_Challenging_Traffic_Conditions_Dataset_for_Large-Range_Vehicle-Infrastructure_ICCV_2025_paper.pdf) | [Project Page](https://advrc-wangbw.github.io/V2XScenes/) | [Access Request](https://advrc-wangbw.github.io/V2XScenes/access/)**

![V2XScenes](opencood/images/v2xscenes_poster.png)

## 📋 Overview

V2XScenes is a comprehensive dataset designed for vehicle-infrastructure cooperative perception research. It features:

- **Multi-modal sensor data**: LiDAR and camera data from both vehicle and roadside platforms
- **Diverse traffic conditions**: Multiple challenging scenarios including various weather and traffic states
- **Large-range coverage**: Data collected over extended road sections for realistic deployment scenarios
- **High-quality annotations**: Carefully labeled 3D bounding boxes and metadata

## 📥 Data Preparation

### Step 1: Request Dataset Access

To obtain the V2XScenes dataset, please visit our [Data Access Portal](https://advrc-wangbw.github.io/V2XScenes/access/) and submit your access request.

### Step 2: Organize Dataset Structure

Create a `dataset` folder under the V2XScenes repository root and organize the downloaded data as follows:

```
dataset/v2xscenes/
├── calibration/                  # Sensor calibration parameters
├── data/
│   └── 20240712_111606_300_1720754373_to_1720754381_8_24/  # Scene data directory
│       ├── label_new/            # Annotation data
│       ├── road_camera/          # Roadside camera data
│       ├── road_lidar/           # Roadside LiDAR data
│       ├── veh_camera/           # Vehicle-mounted camera data
│       ├── veh_lidar/            # Vehicle-mounted LiDAR data
│       ├── visualization/        # Visualization results
│       ├── All_Path_Maps.txt     # Path mapping configuration
│       ├── gps.pkl               # GPS data (pickle format)
│       ├── imu.pkl               # IMU inertial measurement data (pickle format)
│       ├── map.pkl               # Map data (pickle format)
│       ├── odom.pkl              # Odometry data (pickle format)
│       ├── pose.pkl              # Pose data (pickle format)
│       ├── Timestamp.pkl         # Timestamp data (pickle format)
│       └── Timestamp.txt         # Timestamp data (text format)
│   └── ... (additional scenes)
└── tools/                        # Dataset preparation and visualization tools
```

### Step 3: Create Symbolic Links (Optional)

To link data from a custom location, use the provided utility script:

```bash
python dataset/v2xscenes/tools/create_v2x_links.py --target /path/to/your/data
```

## 📦 Installation

### Step 1: Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate v2xscenes

# Install package in development mode
python setup.py develop
```

### Step 2: Install Spconv

We use spconv (1.2.1 or 2.x) for voxel feature generation. **Note**: Checkpoints are stored in spconv 1.2.1 format and are not compatible with 2.x.

**For spconv 2.x** (recommended for easier installation):
Check the [official spconv table](https://github.com/traveller59/spconv#spconv-spatially-sparse-convolution-library) and run the appropriate command for your CUDA version. Example:

```bash
pip install spconv-cu116  # Replace cu116 with your CUDA version (cu111, cu118, etc.)
```

**For spconv 1.2.1** (required if using provided checkpoints):
Follow the [spconv 1.2.1 installation guide](https://github.com/traveller59/spconv/tree/v1.2.1).
Alternatively, refer to the [CoAlign Installation Documentation](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie#doxcn5rISC6NcfXIUnWFnXhTEzd) for detailed setup instructions.

### Step 3: Compile CUDA Kernels

Compile the bounding box IoU CUDA kernels:

```bash
python opencood/utils/setup.py build_ext --inplace
```

## 🚀 Training and Evaluation

### Training

Example training command using the Where2Comm configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_v2xscenes.py \
    -y ./opencood/hypes_yaml/v2xscenes/v2xsences_where2comm.yaml
```

### Visualization and Debugging

To visualize training results and verify label-data alignment:

1. Open `./opencood/data_utils/datasets/intermediate_heter_fusion_dataset_v2xscenes.py`
2. Change `PLOT = False` to `PLOT = True`

Example visualizations:

| Fusion Result | No Fusion Result |
|:---:|:---:|
| ![Fusion](opencood/images/V2xsences_where2comm_all.yaml_1_check_lidar_fusion.png) | ![No Fusion](opencood/images/V2xsences_where2comm_all.yaml_1_check_vehicle_3D.png) |

### Model Inference

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
    --hypes_yaml ${CONFIG_FILE} \
    [--model_dir ${CHECKPOINT_FOLDER} \
    --half]
```

## 📚 Acknowledgements

This project builds upon the excellent work from:
- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) - Collaborative perception framework
- [HEAL](https://github.com/yifanlu0227/HEAL) - Heterogeneous agent learning

## 📖 Citation

If you use V2XScenes in your research, please cite our paper:

```bibtex
@inproceedings{wang2025v2xscenes,
  title={V2XScenes: A Multiple Challenging Traffic Conditions Dataset for Large-Range Vehicle-Infrastructure Collaborative Perception},
  author={Wang, Bowen and Wang, Yafei and Gong, Wei and Chen, Siheng and Liu, Genjia and Xiong, Minhao and Ng, Chin Long},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={28385--28395},
  year={2025}
}
```

## 📄 License

Please refer to the LICENSE file for usage terms and conditions.

## 📞 Contact

For questions or issues regarding the dataset, please visit our [Access Portal](https://advrc-wangbw.github.io/V2XScenes/access/) or open an issue on GitHub.
