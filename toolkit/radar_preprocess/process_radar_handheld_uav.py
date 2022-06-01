import yaml
import inspect
from pcl2depth import velo_points_2_pano
import scipy.io
import numpy as np
import os
from os.path import join
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.insert(1, parentdir)

# get config
with open(join(parentdir, 'config.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

# Select Platform
# platform = 'dataset_creation_handheld'
platform = 'dataset_creation_drone'  # UAV

exp_names = cfg[platform]['all_exp_files']
pendrive_dir = cfg[platform]['dataroot']
v_fov = tuple(map(int, cfg[platform]['pcl2depth']['v_fov'][1:-1].split(',')))
h_fov = tuple(map(int, cfg[platform]['pcl2depth']['h_fov'][1:-1].split(',')))
nb_overlay_frames = cfg[platform]['pcl2depth']['nb_overlay_frames']
save_dir = pendrive_dir

for BAG_DATE in exp_names:
    print('********* Processing {} *********'.format(BAG_DATE))
    ROS_SAVE_DIR = join(save_dir, BAG_DATE)
    MMWAVE_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR,  'mmwave_middle'])
    print(" Creating folder for mmWave depth images {}".format(MMWAVE_SAVE_PATH))
    if not os.path.exists(MMWAVE_SAVE_PATH):
        os.makedirs(MMWAVE_SAVE_PATH)

    MMWAVE_READ_PATH = os.path.join(*[ROS_SAVE_DIR,  'mmwave_middle_pcl'])
    mmwave_file_list = os.listdir(MMWAVE_READ_PATH)
    mmwave_file_list.sort()
    mmwave_ts_list = [int(i[:-4]) for i in mmwave_file_list]

    ###################
    # Read all frames #
    ###################
    frames = []
    for file in mmwave_file_list:
        mat = scipy.io.loadmat(os.path.join(MMWAVE_READ_PATH, file))
        pc = np.array(mat['frame'])
        upper_row_filter = (pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2) ** 0.5 < cfg[platform]['pcl2depth']['mmwave_dist_max']
        lower_row_filter = (pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2) ** 0.5 > cfg[platform]['pcl2depth']['mmwave_dist_min']
        row_filter = np.bitwise_and(upper_row_filter, lower_row_filter)
        frames.append(pc[row_filter, :])

    ###################
    # Overlay frames  #
    ###################
    frames = np.array(frames)
    # overlay frames accounting for sparse pcl
    overlay_frames = list()
    # frames_array = np.array(frames)
    for i in range(frames.shape[0]):
        if i < nb_overlay_frames:
            tmp = frames[i: i + nb_overlay_frames]
        else:
            tmp = frames[i - nb_overlay_frames:i]
        try:
            overlay_frames.append(np.concatenate(tmp))
        except:
            print('error')

    ###################
    # Save Images     #
    ###################
    for timestamp, frame in tqdm(zip(mmwave_ts_list, overlay_frames), total=len(mmwave_ts_list)):
        pano_img = velo_points_2_pano(frame,
                                      cfg[platform]['pcl2depth']['v_res'],
                                      cfg[platform]['pcl2depth']['h_res'],
                                      v_fov,
                                      h_fov,
                                      cfg[platform]['pcl2depth']['max_v'],
                                      depth=True)
        try:
            pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))
            pc_path = os.path.join(MMWAVE_SAVE_PATH, str(timestamp) + '.png')
            cv2.imwrite(pc_path, pano_img)
        except:
            width = (h_fov[1] - h_fov[0] + 2) * 2
            height = (v_fov[1] - v_fov[0] + 2) * 2
            blank_image = np.zeros((height, width), np.uint8)
            pc_path = os.path.join(MMWAVE_SAVE_PATH, str(timestamp) + '.png')
            print('No point in the frame, empty image at: ' + pc_path)
            cv2.imwrite(pc_path, blank_image)
