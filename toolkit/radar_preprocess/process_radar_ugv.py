from pyquaternion import Quaternion
import pptk  # pptk only supports python 3.7 for now
import math
from util.mmwave_bag import make_frames_from_csv_doppler, make_frames_from_csv
import tqdm
import cv2
import collections
from util.pcl2depth import velo_points_2_pano
import csv
from os.path import join
import yaml
import os
import numpy as np
import time
import shutil
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')


def plot_depth(frame_idx, timestamp, frame, map_dir, cfg):
    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))
    # only select those points with the certain range (in meters) - 5 meter for this TI board
    eff_rows_idx = (frame[:, 1] ** 2 + frame[:, 0] ** 2) ** 0.5 < cfg['base_conf']['img_width']
    pano_img = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                  v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

    pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))

    fig_path = join(map_dir, '{}_{}.png'.format(frame_idx, timestamp))

    cv2.imshow("grid", pano_img)
    cv2.waitKey(1)

    cv2.imwrite(fig_path, pano_img)


def quaternion_to_rotation_matrix(quat):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=q.dtype)
    return rot_matrix


def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return np.array([qx, qy, qz, qw])


# get config
project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

fig_dir = cfg['base_conf']['fig_base']
# data_dir = join(cfg['base_conf']['data_base'], 'odom')
data_dir = cfg['base_conf']['pendrive']
exp_names = cfg['pre_process']['prepare_robot']['exp_names']
topics = ['_slash_mmWaveDataHdl_slash_RScan_middle', '_slash_mmWaveDataHdl_slash_RScan_right',
          '_slash_mmWaveDataHdl_slash_RScan_left', '_slash_radar_slash_RScan']
middle_transform = np.array(cfg['radar']['translation_matrix']['middle'])
left_transform = np.array(cfg['radar']['translation_matrix']['left'])
right_transform = np.array(cfg['radar']['translation_matrix']['right'])
mt_matrix = quaternion_to_rotation_matrix(euler_to_quaternion(middle_transform[3], middle_transform[4], middle_transform[5]))
lt_matrix = quaternion_to_rotation_matrix(left_transform[3:7])
rt_matrix = quaternion_to_rotation_matrix(right_transform[3:7])
left_quaternion = Quaternion(axis=[0, 0, 1], angle=math.pi / 2)
right_quaternion = Quaternion(axis=[0, 0, 1], angle=-math.pi / 2)
print(lt_matrix)
print(rt_matrix)

align_interval = 5e7

for exp_name in exp_names:
    # process middle
    topic = '_slash_mmWaveDataHdl_slash_RScan_middle'
    csv_path = join(data_dir, str(exp_name),
                    topic + '.csv')
    print(csv_path)
    if not os.path.exists(csv_path):
        continue

    # readings_dict = make_frames_from_csv_doppler(csv_path)
    readings_dict = make_frames_from_csv(csv_path)

    #!!! sort the dict before using
    data_dict = collections.OrderedDict(sorted(readings_dict.items()))

    frames = list()
    max_pts = 0
    intensities = list()
    doppler = list()
    timestamps = list()
    valid_data = list()
    count = 0
    for timestamp, pts in data_dict.items():
        count = count + 1
        # log to monitor abnormal records
        if len(pts) > max_pts:
            max_pts = len(pts)
        # iterate each pt
        heatmap_per_frame = list()
        test_frame = list()
        for pt in pts:
            tmp = np.array(pt)
            test_frame.append(tmp[[0, 1, 2, 3, 5]])
            tmp_loc = tmp[0:3]
            #tmp_loc = np.concatenate((tmp_loc, [1]))

            if topic == '_slash_mmWaveDataHdl_slash_RScan_middle':
                translated_tmp = tmp_loc + middle_transform[0:3]
            elif topic == '_slash_mmWaveDataHdl_slash_RScan_left':
                translated_tmp = left_quaternion.rotate(tmp_loc) + left_transform[0:3]
            elif topic == '_slash_mmWaveDataHdl_slash_RScan_right':
                translated_tmp = right_quaternion.rotate(tmp_loc) + right_transform[0:3]
            tmp[0] = translated_tmp[0]
            tmp[1] = translated_tmp[1]
            tmp[2] = translated_tmp[2]
            heatmap_per_frame.append(tmp[[0, 1, 2, 3, 5]])
            intensities.append(tmp[3])
            doppler.append(tmp[5])
        # if count == 100:
            # pptk.viewer(np.array(test_frame)[:,0:3])
        # do not add empty frames
        if not heatmap_per_frame:
            continue
        frames.append(np.array(heatmap_per_frame))
        timestamps.append(timestamp)
        valid_data.append([0, 0])

    print('max ppf is {}, length of data is {}'.format(max_pts, len(readings_dict)))
    print('max intensity: {}, min intensity {}'.format(max(intensities), min(intensities)))
    print('doopler: max intensity: {}, min intensity {}'.format(max(doppler), min(doppler)))

    # process left and right
    for topic in ['_slash_mmWaveDataHdl_slash_RScan_left', '_slash_mmWaveDataHdl_slash_RScan_right']:
        csv_path = join(data_dir, str(exp_name),
                        topic + '.csv')
        print(csv_path)
        if not os.path.exists(csv_path):
            continue

        # readings_dict = make_frames_from_csv_doppler(csv_path)
        readings_dict = make_frames_from_csv(csv_path)

        #!!! sort the dict before using
        data_dict = collections.OrderedDict(sorted(readings_dict.items()))

        count = 0
        for timestamp, pts in data_dict.items():
            count = count + 1
            # log to monitor abnormal records
            if len(pts) > max_pts:
                max_pts = len(pts)
            # iterate each pt
            heatmap_per_frame = list()
            test_frame = list()
            for pt in pts:
                tmp = np.array(pt)
                test_frame.append(tmp[[0, 1, 2, 3, 5]])
                tmp_loc = tmp[0:3]
                # tmp_loc = np.concatenate((tmp_loc, [1]))
                if topic == '_slash_mmWaveDataHdl_slash_RScan_middle':
                    translated_tmp = tmp_loc + middle_transform[0:3]
                elif topic == '_slash_mmWaveDataHdl_slash_RScan_left':
                    translated_tmp = left_quaternion.rotate(tmp_loc) + left_transform[0:3]
                elif topic == '_slash_mmWaveDataHdl_slash_RScan_right':
                    translated_tmp = right_quaternion.rotate(tmp_loc) + right_transform[0:3]
                assert(tmp[0] != translated_tmp[0])
                tmp[0] = translated_tmp[0]
                tmp[1] = translated_tmp[1]
                tmp[2] = translated_tmp[2]
                heatmap_per_frame.append(tmp[[0, 1, 2, 3, 5]])
                intensities.append(tmp[3])
                doppler.append(tmp[5])

            # if count == 100:
            #     pptk.viewer(np.array(test_frame)[:,0:3])

            # do not add empty frames
            if not heatmap_per_frame:
                continue

            # align frames
            for i in range(0, len(timestamps)):
                if abs(int(timestamp) - int(timestamps[i])) <= align_interval:
                    frames[i] = np.concatenate((frames[i], np.array(heatmap_per_frame)))
                    if topic == '_slash_mmWaveDataHdl_slash_RScan_left':
                        valid_data[i][0] = 1
                    elif topic == '_slash_mmWaveDataHdl_slash_RScan_right':
                        valid_data[i][1] = 1
    print(valid_data)
    # pptk.viewer(np.array(frames[100])[:,0:3])

    # overlay frames accounting for sparse pcl
    nb_overlay_frames = cfg['pcl2depth']['nb_overlay_frames']
    # frame_buffer_ls = deque(maxlen=nb_overlay_frames)
    overlay_frames = list()
    frames_array = np.array(frames)
    for i in range(frames_array.shape[0]):
        if i < nb_overlay_frames:
            tmp = frames_array[i: i + nb_overlay_frames]
        else:
            tmp = frames_array[i - nb_overlay_frames:i]

        try:
            overlay_frames.append(np.concatenate(tmp))
        except:
            print('error')

    # radar_map_dir = join(data_dir, str(exp_name), 'mmwave_all_clean')
    radar_map_dir = join(data_dir, str(exp_name), 'mmwave_all')

    if os.path.exists(radar_map_dir):
        shutil.rmtree(radar_map_dir)
        time.sleep(5)
        os.makedirs(radar_map_dir)
    else:
        os.makedirs(radar_map_dir)

    # pcl to depth
    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))

    frame_idx = 0
    for timestamp, frame in tqdm.tqdm(zip(timestamps, overlay_frames), total=len(timestamps)):
        if valid_data[frame_idx][0] == 0 or valid_data[frame_idx][1] == 0:
            frame_idx = frame_idx + 1
            continue
        # only select those points with the certain range (in meters) - 5.12 meter for this TI board
        eff_rows_idx = (frame[:, 1] ** 2 + frame[:, 0] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_thre']
        pano_img = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                      v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

        if pano_img.size == 0:
            print('{} frame skipped as all pts are out of fov!'.format(frame_idx))
            frame_idx = frame_idx + 1
            continue

        pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))

        fig_path = join(radar_map_dir, '{}.png'.format(timestamp))

        cv2.imshow("grid", pano_img)
        cv2.waitKey(1)

        cv2.imwrite(fig_path, pano_img)

        frame_idx += 1

    print('In total {} images'.format(frame_idx))
