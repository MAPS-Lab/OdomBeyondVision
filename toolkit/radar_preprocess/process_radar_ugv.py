import yaml
import math
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
from pyquaternion import Quaternion

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.insert(1, parentdir)

# get config
with open(join(parentdir, 'config.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

# Select Platform
platform = 'dataset_creation_robot'  # UGV

exp_names = cfg[platform]['all_exp_files']
pendrive_dir = cfg[platform]['dataroot']
v_fov = tuple(map(int, cfg[platform]['pcl2depth']['v_fov'][1:-1].split(',')))
h_fov = tuple(map(int, cfg[platform]['pcl2depth']['h_multi_fov'][1:-1].split(',')))
nb_overlay_frames = cfg[platform]['pcl2depth']['nb_overlay_frames']
save_dir = pendrive_dir

left_quaternion = Quaternion(axis=[0, 0, 1], angle=math.pi / 2)
right_quaternion = Quaternion(axis=[0, 0, 1], angle=-math.pi / 2)
middle_transform = np.array(cfg[platform]['translation_matrix']['middle'])
left_transform = np.array(cfg[platform]['translation_matrix']['left'])
right_transform = np.array(cfg[platform]['translation_matrix']['right'])


def transform(pc, q, t):
    pc_new = []
    for i in pc:
        i = i[0:3]
        if q:
            translated_i = q.rotate(i) + t[0:3]
        else:
            translated_i = i + t[0:3]
        pc_new.append(translated_i)
    pc_new = np.array(pc_new)
    # print(pc_new.shape)
    return pc_new


def find_nearest(ts_list, target):
    interval = 100000000000
    result = 0
    idx = 0
    for i in range(len(ts_list)):
        ts = ts_list[i]
        tmp = abs(ts - target)
        if tmp < interval:
            result = ts
            interval = tmp
            idx = i
    return result, idx


for BAG_DATE in exp_names:
    print('********* Processing {} *********'.format(BAG_DATE))
    ROS_SAVE_DIR = join(save_dir, BAG_DATE)
    MMWAVE_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR,  'mmwave_all'])
    print(" Creating folder for mmWave depth images {}".format(MMWAVE_SAVE_PATH))
    if not os.path.exists(MMWAVE_SAVE_PATH):
        os.makedirs(MMWAVE_SAVE_PATH)

    MMWAVE_MIDDLE_PATH = os.path.join(*[ROS_SAVE_DIR,  'mmwave_middle_pcl'])
    mmwave_file_list_middle = os.listdir(MMWAVE_MIDDLE_PATH)
    mmwave_file_list_middle.sort()
    mmwave_ts_list_middle = [int(i[:-4]) for i in mmwave_file_list_middle]

    MMWAVE_LEFT_PATH = os.path.join(*[ROS_SAVE_DIR,  'mmwave_left_pcl'])
    mmwave_file_list_left = os.listdir(MMWAVE_LEFT_PATH)
    mmwave_file_list_left.sort()
    mmwave_ts_list_left = [int(i[:-4]) for i in mmwave_file_list_left]

    MMWAVE_RIGHT_PATH = os.path.join(*[ROS_SAVE_DIR,  'mmwave_right_pcl'])
    mmwave_file_list_right = os.listdir(MMWAVE_RIGHT_PATH)
    mmwave_file_list_right.sort()
    mmwave_ts_list_right = [int(i[:-4]) for i in mmwave_file_list_right]

    ###################
    # Read all frames #
    ###################
    frames_middle = []
    for file in mmwave_file_list_middle:
        mat = scipy.io.loadmat(os.path.join(MMWAVE_MIDDLE_PATH, file))
        pc = np.array(mat['frame'])
        upper_row_filter = (pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2) ** 0.5 < cfg[platform]['pcl2depth']['mmwave_dist_max']
        lower_row_filter = (pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2) ** 0.5 > cfg[platform]['pcl2depth']['mmwave_dist_min']
        row_filter = np.bitwise_and(upper_row_filter, lower_row_filter)
        pc = pc[row_filter, :]
        pc = transform(pc, None, middle_transform)
        frames_middle.append(pc)
    frames_left = []
    for file in mmwave_file_list_left:
        mat = scipy.io.loadmat(os.path.join(MMWAVE_LEFT_PATH, file))
        pc = np.array(mat['frame'])
        upper_row_filter = (pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2) ** 0.5 < cfg[platform]['pcl2depth']['mmwave_dist_max']
        lower_row_filter = (pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2) ** 0.5 > cfg[platform]['pcl2depth']['mmwave_dist_min']
        row_filter = np.bitwise_and(upper_row_filter, lower_row_filter)
        pc = pc[row_filter, :]
        pc = transform(pc, left_quaternion, left_transform)
        frames_left.append(pc)
    frames_right = []
    for file in mmwave_file_list_right:
        mat = scipy.io.loadmat(os.path.join(MMWAVE_RIGHT_PATH, file))
        pc = np.array(mat['frame'])
        upper_row_filter = (pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2) ** 0.5 < cfg[platform]['pcl2depth']['mmwave_dist_max']
        lower_row_filter = (pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2) ** 0.5 > cfg[platform]['pcl2depth']['mmwave_dist_min']
        row_filter = np.bitwise_and(upper_row_filter, lower_row_filter)
        pc = pc[row_filter, :]
        pc = transform(pc, right_quaternion, right_transform)
        frames_right.append(pc)

    ###################
    # Stitch frames   #
    ###################
    frames = []
    for middle_idx in range(len(mmwave_ts_list_middle)):
        middle_ts = mmwave_ts_list_middle[middle_idx]
        middle_frame = frames_middle[middle_idx]
        left_match_ts, left_match_idx = find_nearest(mmwave_ts_list_left, middle_ts)
        right_match_ts, right_match_idx = find_nearest(mmwave_ts_list_right, middle_ts)
        left_match_frame = frames_left[left_match_idx]
        right_match_frame = frames_right[right_match_idx]
        frame = np.array([middle_frame, left_match_frame, right_match_frame])
        frames.append(np.concatenate(frame))

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
    for timestamp, frame in tqdm(zip(mmwave_ts_list_middle, overlay_frames), total=len(mmwave_ts_list_middle)):
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
