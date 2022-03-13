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

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
import sys
sys.path.append(parentdir)
from util.pcl2depth import velo_points_2_pano
sys.path.insert(1, parentdir)
# get config
project_dir = os.path.dirname(os.getcwd())

with open(join(parentdir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

bridge = CvBridge()

pendrive_dir = join(cfg['base_conf']['pendrive'])
save_dir = join(os.path.dirname(parentdir), 'data', 'odom')
exp_names = cfg['pre_process']['prepare_handheld']['exp_names']

counter = 0
for BAG_DATE in exp_names:
    print('********* Processing {} *********'.format(BAG_DATE))
    ROS_SAVE_DIR = join(save_dir, BAG_DATE)

    ROSBAG_PATH = os.path.join(pendrive_dir, 'hand_' + BAG_DATE+'.bag')

    RGB_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR,  'rgb'])
    DEPTH_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR, 'depth'])
    THERMAL_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR, 'thermal'])
    UEYE_SAVE_PATH = os.path.join(*[ROS_SAVE_DIR, 'ueye'])

    print(" Saving RGB images into {}".format(RGB_SAVE_PATH))
    print(" Saving Depth images into {}".format(DEPTH_SAVE_PATH))

    bag = rosbag.Bag(ROSBAG_PATH, 'r')

    #########################################
    # process topics based on txt and numbers
    #########################################
    for topic in ['/aft_mapped_to_init', '/aft_mapped_to_init_high_frec']:
        filename = join(ROS_SAVE_DIR, string.replace(topic, '/', '_slash_') + '.csv')
        with open(filename, 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            firstIteration = True  # allows header row
            exist_topic = False
            for subtopic, msg, t in bag.read_messages(topic):  # for each instant in time that has data for topicName
                # parse data from this instant, which is of the form of multiple lines of "Name: value\n"
                #    - put it in the form of a list of 2-element lists
                exist_topic = True
                msgString = str(msg)
                msgList = string.split(msgString, '\n')
                instantaneousListOfData = []
                for nameValuePair in msgList:
                    splitPair = string.split(nameValuePair, ':')
                    for i in range(len(splitPair)):  # should be 0 to 1
                        splitPair[i] = string.strip(splitPair[i])
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

    bag.close()
