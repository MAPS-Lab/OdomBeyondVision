# from pcl2depth import velo_points_2_pano
import sys
import os
import rosbag
from os.path import join
import numpy as np
import cv2
import csv
import yaml
from tqdm import tqdm
import string
import sensor_msgs.point_cloud2
import shutil
import scipy.io as sio
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.insert(1, parentdir)

# get config
with open(join(parentdir, 'config.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

# Select Platform
# platform = 'dataset_creation_robot'  # UGV
# platform = 'dataset_creation_handheld'
platform = 'dataset_creation_drone'  # UAV

pendrive_dir = cfg[platform]['dataroot']
save_dir = pendrive_dir
exp_names = cfg[platform]['all_exp_files']
img_topics = cfg[platform]['img_topics']
pcl_topics = cfg[platform]['pcl_topics']
csv_topics = cfg[platform]['csv_topics']
thermal_16bit = cfg[platform]['thermal_16bit']
# num_radar = len([topic for topic in pcl_topics if 'mmWave' in topic])

counter = 0
for BAG_DATE in exp_names:
    print('********* Processing {} *********'.format(BAG_DATE))
    ROS_SAVE_DIR = join(save_dir, BAG_DATE)

    ROSBAG_PATH = os.path.join(pendrive_dir, BAG_DATE + '.bag')

    RGB_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR,  'rgb'])
    DEPTH_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR, 'depth'])
    THERMAL_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR, 'thermal'])

    print(" Creating folder for RGB images {}".format(RGB_SAVE_PATH))
    print(" Creating folder for Depth images {}".format(DEPTH_SAVE_PATH))
    print(" Creating folder for Thermal images {}".format(THERMAL_SAVE_PATH))

    for path in [RGB_SAVE_PATH, DEPTH_SAVE_PATH, THERMAL_SAVE_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
        # else:
        #     shutil.rmtree(path)
        #     os.makedirs(path)

    print("Reading Rosbag")
    bag = rosbag.Bag(ROSBAG_PATH, 'r')

    #########################################
    # process topics based on txt and numbers
    #########################################
    print("Reading CSV topics")
    for topic in csv_topics:
        filename = join(ROS_SAVE_DIR, str.replace(topic, '/', '_slash_') + '.csv')
        with open(filename, 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            firstIteration = True  # allows header row
            exist_topic = False
            for subtopic, msg, t in bag.read_messages(topic):  # for each instant in time that has data for topicName
                # parse data from this instant, which is of the form of multiple lines of "Name: value\n"
                #    - put it in the form of a list of 2-element lists
                exist_topic = True
                msgString = str(msg)
                msgList = str.split(msgString, '\n')
                instantaneousListOfData = []
                for nameValuePair in msgList:
                    splitPair = str.split(nameValuePair, ':')
                    for i in range(len(splitPair)):  # should be 0 to 1
                        splitPair[i] = str.strip(splitPair[i])
                    instantaneousListOfData.append(splitPair)
                # write the first row from the first element of each pair
                if firstIteration:  # header
                    headers = ["rosbagTimestamp"]  # first column header
                    for pair in instantaneousListOfData:
                        headers.append(pair[0])
                    filewriter.writerow(headers)
                    firstIteration = False
                # write the value from each pair to the file
                values = [str(t)]  # first column will have rosbag timestamp
                for pair in instantaneousListOfData:
                    if len(pair) > 1:
                        values.append(pair[1])
                filewriter.writerow(values)
            if not exist_topic:
                os.remove(filename)

    #########################################
    # process topics based on point cloud
    #########################################
    print("Reading PCL topics")
    # for topic, msg, t in rosbag.Bag(ROSBAG_PATH, 'r').read_messages(topics=['/velodyne_points', '/mmWaveDataHdl/RScan_middle']):
    # for topic, msg, t in rosbag.Bag(ROSBAG_PATH, 'r').read_messages(topics=pcl_topics):
    for topic, msg, t in tqdm(bag.read_messages(topics=pcl_topics)):
        # init a directory
        if 'ti_mmwave' in topic:
            pcl_dir = join(ROS_SAVE_DIR, 'mmwave_middle_pcl')
            # depth_dir = join(ROS_SAVE_DIR, 'mmwave_middle')
        elif 'velodyne' in topic:
            pcl_dir = join(ROS_SAVE_DIR, 'lidar_pcl')
            # depth_dir = join(ROS_SAVE_DIR, 'lidar')
        else:
            pcl_dir = join(ROS_SAVE_DIR, 'mmwave_' + topic.split('_')[-1] + '_pcl')
            # depth_dir = join(ROS_SAVE_DIR, 'mmwave_' + topic.split('_')[-1])

        if not os.path.exists(pcl_dir):
            os.makedirs(pcl_dir)
        # if not os.path.exists(depth_dir):
        #     os.makedirs(depth_dir)

        if msg.header.stamp.to_nsec() == 0:
            ts = t.to_nsec()  # Using bag timestamp
        else:
            ts = msg.header.stamp.to_nsec()  # Using msg timestamp

        # read point cloud to array
        pc = [point for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True)]
        pc = np.array(pc)

        # save to mat
        filename = str(ts) + '.mat'
        filepath = join(pcl_dir, filename)
        sio.savemat(filepath, mdict={'frame': pc})

        # # save to img
        # v_fov = tuple(map(int, cfg[platform]['pcl2depth']['v_fov'][1:-1].split(',')))
        # h_fov = tuple(map(int, cfg[platform]['pcl2depth']['h_fov'][1:-1].split(',')))
        # upper_row_filter = (pc[:, 1] ** 2 + pc[:, 0] ** 2) ** 0.5 < cfg[platform]['pcl2depth']['mmwave_dist_max']
        # lower_row_filter = (pc[:, 1] ** 2 + pc[:, 0] ** 2) ** 0.5 > cfg[platform]['pcl2depth']['mmwave_dist_min']
        # row_filter = np.bitwise_and(upper_row_filter, lower_row_filter)
        # pc_filtered = pc[row_filter, :]
        # if num_radar > 1:
        #     mmwave_buffer[topic.split('/')[-1]].append(pc)
        #     ts_buffer[topic.split('/')[-1]].append(ts)
        # pano_img = velo_points_2_pano(pc_filtered,
        #                               cfg[platform]['pcl2depth']['v_res'],
        #                               cfg[platform]['pcl2depth']['h_res'],
        #                               v_fov,
        #                               h_fov,
        #                               cfg[platform]['pcl2depth']['max_v'],
        #                               depth=True)
        # if pano_img.size == 0:
        #     print('The frame skipped as all pts are out of fov!')
        #     continue
        # pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))
        # pc_name = str(ts) + ".png"
        # pc_path = os.path.join(depth_dir, pc_name)
        # cv2.imwrite(pc_path, pano_img)

    #########################################
    # process topics based on images
    #########################################
    print("Reading Image topics")
    # for topic, msg, t in tqdm(rosbag.Bag(ROSBAG_PATH, 'r').read_messages(topics=img_topics)):
    for topic, msg, t in tqdm(bag.read_messages(topics=img_topics)):
        if topic == '/camera/color/image_raw':
            counter += 1
            #image_name = 'color_'+str(msg.header.stamp.secs)+ '.' + "{0:09d}".format(msg.header.stamp.nsecs) + ".png"
            image_name = str(t) + ".png"
            np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)  # 8bit RGB image
            cv_image = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(RGB_SAVE_PATH, image_name), cv_image)
        if topic == '/camera/depth/image_rect_raw':
            counter += 1
            # image_name = 'depth_'+str(msg.header.stamp.secs)+ '.' + "{0:09d}".format(msg.header.stamp.nsecs) + ".png"
            image_name = str(t) + ".png"
            np_arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)  # 16bit depth image
            cv2.imwrite(os.path.join(DEPTH_SAVE_PATH, image_name), np_arr)
        if topic == '/flir_boson/image_raw':
            counter += 1
            # image_name = 'depth_'+str(msg.header.stamp.secs)+ '.' + "{0:09d}".format(msg.header.stamp.nsecs) + ".png"
            image_name = str(t) + ".png"
            if thermal_16bit:
                np_arr = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width, -1)  # 16bit thermal image
            else:
                np_arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)  # 8bit thermal image
            np_arr = np.reshape(np_arr, (msg.height, msg.width))
            # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(THERMAL_SAVE_PATH, image_name), np_arr)

    bag.close()
