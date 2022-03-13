from util.mmwave_bag import make_frames_from_csv_doppler
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
    h_fov = tuple(map(int, cfg['pcl2depth']['h_fov'][1:-1].split(',')))
    # only select those points with the certain range (in meters) - 5 meter for this TI board
    eff_rows_idx = (frame[:, 1] ** 2 + frame[:, 0] ** 2) ** 0.5 < cfg['base_conf']['img_width']
    pano_img = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                  v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

    pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))

    fig_path = join(map_dir, '{}_{}.png'.format(frame_idx, timestamp))

    cv2.imshow("grid", pano_img)
    cv2.waitKey(1)

    cv2.imwrite(fig_path, pano_img)


# get config
project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

fig_dir = cfg['base_conf']['fig_base']
data_dir = join(cfg['base_conf']['data_base'], 'odom')
exp_names = cfg['pre_process']['prepare_robot']['exp_names']
topics = ['_slash_mmWaveDataHdl_slash_RScan_middle',
          '_slash_mmWaveDataHdl_slash_RScan_right',
          '_slash_mmWaveDataHdl_slash_RScan_left',
          '_slash_radar_slash_RScan']

for exp_name in exp_names:
    for topic in topics:
        csv_path = join(data_dir, str(exp_name),
                        topic + '.csv')
        print(csv_path)
        if not os.path.exists(csv_path):
            continue

        readings_dict = make_frames_from_csv_doppler(csv_path)
        #!!! sort the dict before using
        data_dict = collections.OrderedDict(sorted(readings_dict.items()))

        frames = list()
        max_pts = 0
        intensities = list()
        doppler = list()
        timestamps = list()

        for timestamp, pts in data_dict.items():
            # log to monitor abnormal records
            if len(pts) > max_pts:
                max_pts = len(pts)
            # iterate each pt
            heatmap_per_frame = list()
            for pt in pts:
                tmp = np.array(pt)
                heatmap_per_frame.append(tmp[[0, 1, 2, 3, 5]])
                intensities.append(tmp[3])
                doppler.append(tmp[5])
            # do not add empty frames
            if not heatmap_per_frame:
                continue
            frames.append(np.array(heatmap_per_frame))
            timestamps.append(timestamp)

        print('max ppf is {}, length of data is {}'.format(max_pts, len(readings_dict)))
        print('max intensity: {}, min intensity {}'.format(max(intensities), min(intensities)))
        print('doopler: max intensity: {}, min intensity {}'.format(max(doppler), min(doppler)))

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

        if topic.split('_')[-1] == 'RScan':
            radar_map_dir = join(data_dir, str(exp_name), 'mmwave_middle')
        else:
            radar_map_dir = join(data_dir, str(exp_name), 'mmwave_' + topic.split('_')[-1])

        if os.path.exists(radar_map_dir):
            shutil.rmtree(radar_map_dir)
            time.sleep(5)
            os.makedirs(radar_map_dir)
        else:
            os.makedirs(radar_map_dir)

        # pcl to depth
        v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
        h_fov = tuple(map(int, cfg['pcl2depth']['h_fov'][1:-1].split(',')))

        frame_idx = 0
        for timestamp, frame in tqdm.tqdm(zip(timestamps, overlay_frames), total=len(timestamps)):
            # only select those points with the certain range (in meters) - 5.12 meter for this TI board
            eff_rows_idx = (frame[:, 1] ** 2 + frame[:, 0] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_thre']
            pano_img = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                          v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

            if pano_img.size == 0:
                print('{} frame skipped as all pts are out of fov!'.format(frame_idx))
                continue

            pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))

            fig_path = join(radar_map_dir, '{}.png'.format(timestamp))

            cv2.imshow("grid", pano_img)
            cv2.waitKey(1)

            cv2.imwrite(fig_path, pano_img)

            frame_idx += 1

        print('In total {} images'.format(frame_idx))
