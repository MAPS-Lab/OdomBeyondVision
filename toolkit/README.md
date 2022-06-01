# Dataset Toolkit

## Extract files from Rosbags

`extract_rosbag` contains scripts to extract files from rosbags. ROS topics are extracted as image, mat or csv files.

To extract the files, go to the folder and run:
```
python extract_files.py
```

Data storage paths can be configured in 'config.yaml'. Platform can be selected in `extract_files.py` file itself.

## Radar pre-processing

`radar_preprocess` folder contains scripts to process radar data for milliEgo training.

The scripts do the following things:
- Overlay radar data. Overlays 3 adjacent radar frames (2 frames in the case of UAV) into 1 frame, in order to increase density and reduce fluctuation.
- Stitching 3 radars on UGV platform.
- Save radar data in the form of depth images

To pre-process the radar data from each platform, run:

```
python process_radar_handheld_uav.py
```
Or:
```
python process_radar_handheld.py
```

Platform can be selected by configuring inside of `process_radar_handheld_uav.py`.

## Create training datasets

We package sequences into h5 files in order to facilitate training.

`create_dataset_milliego` creates training dataset for milliEgo on Handheld and UAV platforms, where there's only 1 radar.

`create_dataset_milliego_ugv` creates training dataset for milliEgo on UGV platform. It uses data from 3 radars.

`create_dataset_deeptio` creates dataset for DeepTIO on Handheld and UGV platforms, where thermal images are 16bit.

`create_dataset_deeptio_uav` creates dataset for DeepTIO on UAV platform, where thermal images are 8bit, and RGB images need to be reshaped.

The scripts do the following things:
- Normalize image data
- Align different modalities by associating timestamps
- Package data into h5 files

Configure the parameters in `config.yaml`.

To create the datasets, simply go to corresponding folder, and run:
```
python os_create_dataset.py
```
