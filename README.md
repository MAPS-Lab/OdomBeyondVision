# OdomBeyondVision: An Indoor Multi-modal Multi-platform Odometry Dataset Beyond the Visible Spectrum

<!-- [<img src="https://i.imgur.com/i0Ko1ze.png" align="center" width="400">](https://youtu.be/EbjNoZcZpzQ) -->

Watch the video:
[<p align="center"><img width="600" src="https://i.imgur.com/i0Ko1ze.png"></p>](https://youtu.be/EbjNoZcZpzQ)


## Abstract

We present the **OdomBeyondVision** dataset, a comprehensive indoor odometry dataset containing the data from both emerging and traditional navigation sensors on diverse platforms. The dataset features:

- The first indoor dataset to simultaneously cover the mmWave and LWIR thermal spectrum.

- \>10km sequences on UGV, UAV and handheld platforms in various indoor scenes and illumination conditions.

- Extrinsic calibration parameters for all platforms and intrinsic parameters for all optic sensors, accompanied by several plug-and-play codes for developing multimodal odometry systems with our dataset.

## Dataset

The full dataset in the format of Rosbags can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1cJ4w3Dj21EMIox2ZSx7TSdzflxNHhTVL?usp=sharing).

Information for these sequences is recorded on [Google Sheet](https://docs.google.com/spreadsheets/d/1n2KkD_vjE7b5-2_qaq-sq0c-cB4JdQMBo3zMRFuTT_w/edit?usp=sharing).

**Relevant ROS topics**

```
/camera/color/image_raw: RGB image from D435/L515
/camera/depth/image_rect_raw: depth image from D435/L515
/flir_boson/image_raw: thermal image from FLIR Boson
/velodyne_points: point cloud data from Velodyne LiDAR
/imu/data: IMU data from Xsnes MTi
/odom: wheel odometry on UGV
/mmWaveDataHdl/RScan_left_scan: left radar on UGV
/mmWaveDataHdl/RScan_middle_scan: central radar on UGV or Handheld
/mmWaveDataHdl/RScan_right_scan: right radar on UGV
/vrpn_client_node/X500_pglite/pose: Vicon pose on UAV
/ti_mmwave/radar_scan_pcl_0: radar on UAV
```

## Toolkit

Dataset toolkits are provided, to:
- Extract raw data from rosbags;
- Preprocess radar data fro milliEgo training;
- Package training data for milliEgo and DeepTIO.

Details can be seen [here](./toolkit/README.md).

## Calibration

Calibration results for extrinsic parameters and intrinsic parameters are provided [here](./calibration/calibration.pdf).

## Benchmark

Training, testing and evaluation for milliEgo and DeepTIO are provided [here](./benchmark/).

Pre-trained models for milliEgo and DeepTIO on all 3 platforms can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1cJ4w3Dj21EMIox2ZSx7TSdzflxNHhTVL?usp=sharing).
