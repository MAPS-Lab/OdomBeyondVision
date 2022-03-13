from pcl2depth import velo_points_2_pano
import sys
import os
import rosbag
from os.path import join
import numpy as np
import cv2
import csv
import yaml
from cv_bridge import CvBridge, CvBridgeError
from tqdm import tqdm
import string
import sensor_msgs.point_cloud2
import shutil
import scipy.io as sio
from tqdm import tqdm

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.insert(1, parentdir)
# get config
project_dir = os.path.dirname(os.getcwd())

with open(join(parentdir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

bridge = CvBridge()

pendrive_dir = join(cfg['base_conf']['pendrive'])
# save_dir = join(os.path.dirname(parentdir), 'data', 'odom')
save_dir = pendrive_dir
# exp_names = cfg['pre_process']['prepare_robot']['exp_names']
# exp_names = cfg['pre_process']['prepare_drone']['exp_names']
exp_names = cfg['pre_process']['prepare_handheld']['exp_names']

counter = 0
for BAG_DATE in exp_names:
    print('********* Processing {} *********'.format(BAG_DATE))
    ROS_SAVE_DIR = join(save_dir, BAG_DATE)

    ROSBAG_PATH = os.path.join(pendrive_dir, BAG_DATE + '.bag')

    RGB_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR,  'rgb'])
    DEPTH_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR, 'depth'])
    THERMAL_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR, 'thermal'])
    MMWAVE_MAT_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR, 'mmwave_mat'])
    MMWAVE_PNG_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR, 'mmwave'])

    print(" Saving RGB images into {}".format(RGB_SAVE_PATH))
    print(" Saving Depth images into {}".format(DEPTH_SAVE_PATH))

    # if not os.path.exists(DEPTH_SAVE_PATH):
    #     os.makedirs(DEPTH_SAVE_PATH)
    # else:
    #     shutil.rmtree(DEPTH_SAVE_PATH)
    #     os.makedirs(DEPTH_SAVE_PATH)
    # if not os.path.exists(RGB_SAVE_PATH):
    #     os.makedirs(RGB_SAVE_PATH)
    # else:
    #     shutil.rmtree(RGB_SAVE_PATH)
    #     os.makedirs(RGB_SAVE_PATH)
    # if not os.path.exists(THERMAL_SAVE_PATH):
    #     os.makedirs(THERMAL_SAVE_PATH)

    for path in [RGB_SAVE_PATH, DEPTH_SAVE_PATH, THERMAL_SAVE_PATH, MMWAVE_MAT_SAVE_PATH, MMWAVE_PNG_SAVE_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
    #     else:
    #         shutil.rmtree(path)
    #         os.makedirs(path)

    bag = rosbag.Bag(ROSBAG_PATH, 'r')

    #########################################
    # process topics based on SYNC MESSAGES
    #########################################
    # for topic in ['/ros_synchronizer/sync_input']:
    #     filename = join(ROS_SAVE_DIR, str.replace(topic, '/', '_slash_') + '.csv')
    #     with open(filename, 'w+') as csvfile:
    #         filewriter = csv.writer(csvfile, delimiter=',')
    #         firstIteration = True  # allows header row
    #         exist_topic = False
    #         for subtopic, msg, t in bag.read_messages(topic):
    #             exist_topic = True
    #             counter += 1
    #             # Parse IMU data
    #             imu_data = np.fromstring(msg.imu_data[1:-1], dtype=np.float64, sep=' ')
    #             imuList = list(imu_data)
    #             # Parse overlayed mmWave PointCloud data
    #             msgList = list(msg.pcl_data.data)
    #             # Save thermal image
    #             image_name = str(t) + ".png"
    #             np_arr = np.fromstring(msg.img_data.data, np.uint16)
    #             np_arr = np.reshape(np_arr, (msg.img_data.height, msg.img_data.width))
    #             cv_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    #             cv2.imwrite(os.path.join(THERMAL_SAVE_PATH, image_name), np_arr)
    #
    #             values = [str(t)]
    #             values.append(str(imuList))
    #             values.append(str(msgList))
    #             filewriter.writerow(values)
    #
    #         if not exist_topic:
    #             os.remove(filename)

    # bag.close()

    #########################################
    # process topics based on txt and numbers
    #########################################
    # for topic in ['/imu/data',
    #               '/vrpn_client_node/X500_pglite/pose',
    #               # '/tf',
    #               # '/ti_mmwave/radar_scan',
    #               # '/ti_mmwave/radar_scan_pcl_0',
    #               ]:
    #     filename = join(ROS_SAVE_DIR, str.replace(topic, '/', '_slash_') + '.csv')
    #     with open(filename, 'w+') as csvfile:
    #         filewriter = csv.writer(csvfile, delimiter=',')
    #         firstIteration = True  # allows header row
    #         exist_topic = False
    #         for subtopic, msg, t in bag.read_messages(topic):  # for each instant in time that has data for topicName
    #             # parse data from this instant, which is of the form of multiple lines of "Name: value\n"
    #             #    - put it in the form of a list of 2-element lists
    #             exist_topic = True
    #             msgString = str(msg)
    #             msgList = str.split(msgString, '\n')
    #             instantaneousListOfData = []
    #             for nameValuePair in msgList:
    #                 splitPair = str.split(nameValuePair, ':')
    #                 for i in range(len(splitPair)):  # should be 0 to 1
    #                     splitPair[i] = str.strip(splitPair[i])
    #                 instantaneousListOfData.append(splitPair)
    #             # write the first row from the first element of each pair
    #             if firstIteration:  # header
    #                 headers = ["rosbagTimestamp"]  # first column header
    #                 for pair in instantaneousListOfData:
    #                     headers.append(pair[0])
    #                 filewriter.writerow(headers)
    #                 firstIteration = False
    #             # write the value from each pair to the file
    #             values = [str(t)]  # first column will have rosbag timestamp
    #             for pair in instantaneousListOfData:
    #                 if len(pair) > 1:
    #                     values.append(pair[1])
    #             filewriter.writerow(values)
    #         if not exist_topic:
    #             os.remove(filename)
    #
    # bag.close()

    #########################################
    # process topics based on point cloud
    #########################################
    # count = 0
    # for topic, msg, t in rosbag.Bag(ROSBAG_PATH, 'r').read_messages(topics=['/ti_mmwave/radar_scan_pcl_0']):
    #     # init a directory
    #     pcl_map_dir = join(ROS_SAVE_DIR, 'mmwave_mat')
    #
    #     if not os.path.exists(pcl_map_dir):
    #         os.makedirs(pcl_map_dir)
    #
    #     pc = [point for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True)]
    #     pc = np.array(pc)
    #
    #     # filename = str(msg.header.stamp.secs) + "{0:09d}".format(msg.header.stamp.nsecs) + '.mat'
    #     filename = str(t) + '.mat'
    #     count += 1
    #     filepath = join(pcl_map_dir, filename)
    #     sio.savemat(filepath, mdict={'frame': pc})
    #
    #     pano_img = velo_points_2_pano(pc, v_res=0.42, h_res=0.35, v_fov=(-15, 15), h_fov=(-55, 55), max_v=6, depth=True)
    #     v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    #     h_fov = tuple(map(int, cfg['pcl2depth']['h_fov'][1:-1].split(',')))
    #     eff_rows_idx = (pc[:, 1] ** 2 + pc[:, 0] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_thre']
    #
    #     pano_img = velo_points_2_pano(pc[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
    #                                   v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)
    #     if pano_img.size == 0:
    #         print('The frame skipped as all pts are out of fov!')
    #         continue
    #
    #     pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))
    #     #     print(pano_img)
    #     # pano_img = velo_points_2_pano(pc, v_res=2, h_res=2, v_fov=(-15,15), h_fov=(-55,55), min_v =0.1, max_v = 6, depth=True)
    #     # pc_name = str(msg.header.stamp.secs) + "{0:09d}".format(msg.header.stamp.nsecs) + ".png"
    #     pc_path = os.path.join(pcl_map_dir, str(t) + '.png')
    #     cv2.imwrite(pc_path, pano_img)
    #
    # bag.close()

    #########################################
    # process topics based on images
    #########################################
    all_mmwave_frames = []
    timestamps = []
    for topic, msg, t in tqdm(rosbag.Bag(ROSBAG_PATH, 'r').read_messages(topics=[
        # '/camera/color/image_raw',
        # '/camera/depth/image_rect_raw',
        # '/flir_boson/image_raw',
        '/mmWaveDataHdl/RScan_middle'
        # '/ti_mmwave/radar_scan_pcl_0'
    ])):
        # if topic == '/camera/color/image_raw':
        #     counter += 1
        #     # image_name = 'color_'+str(msg.header.stamp.secs)+ '.' + "{0:09d}".format(msg.header.stamp.nsecs) + ".png"
        #     image_name = str(t) + ".png"
        #
        #     # np_arr = np.fromstring(msg.data, np.uint8)
        #     # np_arr = bridge.imgmsg_to_cv2(msg, "bgr8")
        #     np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        #     # np_arr = np.reshape(np_arr, (msg.height, msg.width, 3))
        #     # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #     cv_image = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(os.path.join(RGB_SAVE_PATH, image_name), cv_image)
        # if topic == '/camera/depth/image_rect_raw':
        #     counter += 1
        #     # image_name = 'depth_'+str(msg.header.stamp.secs)+ '.' + "{0:09d}".format(msg.header.stamp.nsecs) + ".png"
        #     image_name = str(t) + ".png"
        #     # np_arr = np.fromstring(msg.data, np.uint8)
        #     # np_arr = bridge.imgmsg_to_cv2(msg)
        #     # np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        #     np_arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
        #     # np_arr = np.fromstring(msg.data, np.uint16)
        #     # np_arr = np.reshape(np_arr, (msg.height, msg.width))
        #     # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        #     # cv_image = cv2.cvtColor(np_arr, cv2.COLOR_RGB2GRAY)
        #     # csv_name = str(t) + ".csv"
        #     # np.savetxt(os.path.join(DEPTH_SAVE_PATH, csv_name), np_arr, delimiter=",")
        #     cv2.imwrite(os.path.join(DEPTH_SAVE_PATH, image_name), np_arr)
        # if topic == '/flir_boson/image_raw':
        #     counter += 1
        #     # image_name = 'depth_'+str(msg.header.stamp.secs)+ '.' + "{0:09d}".format(msg.header.stamp.nsecs) + ".png"
        #     image_name = str(t) + ".png"
        #     # np_arr = np.fromstring(msg.data, np.uint8)
        #     # np_arr = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width, -1)
        #     np_arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        #     np_arr = np.reshape(np_arr, (msg.height, msg.width))
        #     # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        #     cv2.imwrite(os.path.join(THERMAL_SAVE_PATH, image_name), np_arr)
        # if topic == '/ti_mmwave/radar_scan_pcl_0':
        if topic == '/mmWaveDataHdl/RScan_middle':
            # # init a directory
            # pcl_map_dir = join(ROS_SAVE_DIR, 'mmwave_mat')
            #
            # if not os.path.exists(pcl_map_dir):
            #     os.makedirs(pcl_map_dir)
            #
            # pc = [point for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True)]
            # pc = np.array(pc)
            #
            # # filename = str(msg.header.stamp.secs) + "{0:09d}".format(msg.header.stamp.nsecs) + '.mat'
            # filename = str(t) + '.mat'
            # filepath = join(pcl_map_dir, filename)
            # sio.savemat(filepath, mdict={'frame': pc})
            #
            # pano_img = velo_points_2_pano(pc, v_res=0.42, h_res=0.35, v_fov=(-15, 15), h_fov=(-55, 55), max_v=6, depth=True)
            # v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
            # h_fov = tuple(map(int, cfg['pcl2depth']['h_fov'][1:-1].split(',')))
            # upper_row_filter = (pc[:, 1] ** 2 + pc[:, 0] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_max']
            # lower_row_filter = (pc[:, 1] ** 2 + pc[:, 0] ** 2) ** 0.5 > cfg['pcl2depth']['mmwave_dist_min']
            # row_filter = np.bitwise_and(upper_row_filter, lower_row_filter)
            #
            # pano_img = velo_points_2_pano(pc[row_filter, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
            #                               v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)
            # if pano_img.size == 0:
            #     print('The frame skipped as all pts are out of fov!')
            #     continue
            #
            # pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))
            # #     print(pano_img)
            # # pano_img = velo_points_2_pano(pc, v_res=2, h_res=2, v_fov=(-15,15), h_fov=(-55,55), min_v =0.1, max_v = 6, depth=True)
            # # pc_name = str(msg.header.stamp.secs) + "{0:09d}".format(msg.header.stamp.nsecs) + ".png"
            # pcl_map_dir = join(ROS_SAVE_DIR, 'mmwave')
            # pc_path = os.path.join(pcl_map_dir, str(t) + '.png')
            # cv2.imwrite(pc_path, pano_img)

            #########################################
            pc = [point for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True)]
            pc = np.array(pc)
            v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
            h_fov = tuple(map(int, cfg['pcl2depth']['h_fov'][1:-1].split(',')))
            upper_row_filter = (pc[:, 1] ** 2 + pc[:, 0] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_max']
            lower_row_filter = (pc[:, 1] ** 2 + pc[:, 0] ** 2) ** 0.5 > cfg['pcl2depth']['mmwave_dist_min']
            row_filter = np.bitwise_and(upper_row_filter, lower_row_filter)
            all_mmwave_frames.append(pc[row_filter, :])
            timestamps.append(t)

    all_mmwave_frames = np.array(all_mmwave_frames)
    # overlay frames accounting for sparse pcl
    nb_overlay_frames = cfg['pcl2depth']['nb_overlay_frames']
    # frame_buffer_ls = deque(maxlen=nb_overlay_frames)
    overlay_frames = list()
    # frames_array = np.array(frames)
    for i in range(all_mmwave_frames.shape[0]):
        if i < nb_overlay_frames:
            tmp = all_mmwave_frames[i: i + nb_overlay_frames]
        else:
            tmp = all_mmwave_frames[i - nb_overlay_frames:i]
        try:
            overlay_frames.append(np.concatenate(tmp))
        except:
            print('error')

    pcl_map_dir = join(ROS_SAVE_DIR, 'mmwave_middle')
    if not os.path.exists(pcl_map_dir):
        os.makedirs(pcl_map_dir)

    for timestamp, frame in tqdm(zip(timestamps, overlay_frames), total=len(timestamps)):
        pano_img = velo_points_2_pano(frame, cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                      v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)
        pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))
        pc_path = os.path.join(pcl_map_dir, str(timestamp) + '.png')
        cv2.imwrite(pc_path, pano_img)

    bag.close()

    #########################################
    # process thermal images
    #########################################
    # for topic, msg, t in tqdm(rosbag.Bag(ROSBAG_PATH, 'r').read_messages(topics=['/flir_boson/image_raw'])):
    #     if topic == '/flir_boson/image_raw':
    #         counter += 1
    #         # image_name = 'depth_'+str(msg.header.stamp.secs)+ '.' + "{0:09d}".format(msg.header.stamp.nsecs) + ".png"
    #         image_name = str(t) + ".png"
    #         # np_arr = np.fromstring(msg.data, np.uint8)
    #         np_arr = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width, -1)
    #         np_arr = np.reshape(np_arr, (msg.height, msg.width))
    #         # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    #         cv2.imwrite(os.path.join(THERMAL_SAVE_PATH, image_name), np_arr)
    #
    # bag.close()
