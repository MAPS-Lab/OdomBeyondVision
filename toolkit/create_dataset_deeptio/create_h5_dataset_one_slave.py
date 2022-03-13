#!/usr/bin/env python
"""
Generate database file in .h5 containing master, slaves sensor, and pseudo groundtruth.
The final output fps depends on the sampling rate input.
"""

import os, os.path
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import argparse
import math
import numpy as np
import h5py
from scipy import misc
import csv
from eulerangles import mat2euler, euler2mat
import yaml
from os.path import join, dirname


SCALER = 1.0 # scale label: 1, 100, 10000
RADIUS_2_DEGREE = 180.0 / math.pi

def rotated_to_local(T_w_c):
    # Input is 7 DoF absolute poses (3 trans, 4 quat), output is 6 DoF relative poses
    poses_local = []
    # T_w_c = np.insert(T_w_c, 0, 1, axis=1) # add dummy timestamp
    for i in range(1, len(T_w_c)):
        T_w_c_im1 = transform44(T_w_c[i-1])
        T_w_c_i = transform44(T_w_c[i])

        T_c_im1_c_i = np.dot(np.linalg.pinv(T_w_c_im1), T_w_c_i)

        # 3D: x, y, z, roll, pitch, yaw
        eular_c_im1_c_i = mat2euler(T_c_im1_c_i[0:3, 0:3])
        poses_local.append([SCALER * T_c_im1_c_i[0, 3], SCALER * T_c_im1_c_i[1, 3], SCALER * T_c_im1_c_i[2, 3],
                            SCALER * eular_c_im1_c_i[2] * RADIUS_2_DEGREE, SCALER * eular_c_im1_c_i[1] * RADIUS_2_DEGREE,
                            SCALER * eular_c_im1_c_i[0] * RADIUS_2_DEGREE])
    poses_local = np.array(poses_local)
    return poses_local


def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    _EPS = np.finfo(float).eps * 4.0
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
            (1.0, 0.0, 0.0, t[0])
            (0.0, 1.0, 0.0, t[1])
            (0.0, 0.0, 1.0, t[2])
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)


def iround(x):
    """iround(number) -> integer
    Round a number to the nearest integer."""
    y = round(x) - .5
    return int(y) + (y > 0)

def main():
    print('WE HAVENT MODIFIED THIS!!!!')
    print('FOR **Master and 2 slaves** ONLY!')

    DESCRIPTION = """This script receives a working directory and a dataset mean for each modality."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataroot', required=True, help='''Specify the dataroot directory.''')
    parser.add_argument('--ref_file_name', required=True,
                        help='''Specify the reference (synchronized) filename to load the data''')
    parser.add_argument('--master', required=True, help='''Specify the master.''')
    parser.add_argument('--slave_1', required=True, help='''Specify the slave_1.''')
    parser.add_argument('--slave_2', required=True, help='''Specify the slave_2.''')
    parser.add_argument('--mean_master_file', required=True, help='''Specify the dataset mean for master.''')
    parser.add_argument('--mean_slave1_file', help='''Specify the dataset mean for slave 1.''')
    parser.add_argument('--range_master_file', required=True, help='''Specify the range file for master.''')
    parser.add_argument('--range_slave1_file', required=True, help='''Specify the range file for slave 1.''')
    parser.add_argument('--save_dir', help='''Specify save directory.''')
    parser.add_argument('--gap', required=True, help='''Specify the sampling gap.''')

    args = parser.parse_args()

    dataroot = args.dataroot
    save_dir = args.save_dir
    ref_file_name = args.ref_file_name
    master = args.master
    slave_1 = args.slave_1
    slave_2 = args.slave_2
    mean_master_file = args.mean_master_file
    mean_slave1_file = args.mean_slave1_file
    range_master_file = args.range_master_file
    range_slave1_file = args.range_slave1_file
    GAP = int(args.gap)

    parent_dir = dirname(dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    with open(join(parent_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    all_exps = cfg['dataset_creation']['all_exp_files']


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_mean_master = open(mean_master_file, "r")
    mean_master_str = file_mean_master.readlines()[0]
    file_mean_master.close()

    file_mean_slave_1 = open(mean_slave1_file, "r")
    mean_slave_1_str = file_mean_slave_1.readlines()[0]
    file_mean_slave_1.close()

    file_range_master = open(range_master_file, "r")
    range_master_str = file_range_master.readlines()[0]
    file_range_master.close()

    file_range_slave_1 = open(range_slave1_file, "r")
    range_slave_1_str = file_range_slave_1.readlines()[0]
    file_range_slave_1.close()

    # IMPORTANT, PLEASE SPECIFY THE SAMPLING RATE/GAP
    odom_data_GAP = [GAP] * len(all_exps)

    seq_counter = 1
    total_img_counter = 0

    # for exp_file in all_exps:
    for j in range(len(all_exps)):
        # img_dir = join(dataroot, exp_file, data_type)
        master_dir = join(dataroot, all_exps[j], master)
        slave_1_dir = join(dataroot, all_exps[j], slave_1)
        file_full_path = join(dataroot, all_exps[j], ref_file_name)
        with open(file_full_path, 'r') as the_files:
            file_lines = [line for line in the_files]

        # Sampling file based on the specified gap
        sampled_files = []
        sampling = odom_data_GAP[j]
        for k in range(0, np.size(file_lines), sampling):
            sampled_files.append(file_lines[k])

        # Variables to save data
        train_timestamp = []
        train_master = [] # thermal/mmwave/depth
        train_slave_1 = [] # rgb
        train_slave_2 = np.empty((len(sampled_files), 20, 6), dtype=np.float64) # imu
        train_label = []

        # save timestamp
        timestamp = [line[:-1].split(',')[2] for line in sampled_files]
        print('Total timestamp: ', np.shape(timestamp))
        train_timestamp.append(timestamp)

        gt_lines_float = []
        for line in sampled_files:
            gt_lines_float.append(np.array(
                [float(line[:-1].split(',')[2]),  # timestamp
                 float(line[:-1].split(',')[3]), float(line[:-1].split(',')[4]), float(line[:-1].split(',')[5]),
                 # translation
                 float(line[:-1].split(',')[6]), float(line[:-1].split(',')[7]),
                 float(line[:-1].split(',')[8]), float(line[:-1].split(',')[9])]))  # quaternion

        lidar_rel_poses = rotated_to_local(gt_lines_float)
        train_label.append(lidar_rel_poses)
        print('GT size: ', np.shape(train_label))

        for k in range(0, len(sampled_files)):

            # read master corresponding to pose
            min_range_master = float(range_master_str.split(',')[0])
            max_range_master = float(range_master_str.split(',')[1])
            master_path = master_dir + '/' + sampled_files[k].split(',')[0]  # idx 0 is always for the master!
            # normalize master image
            master_img = misc.imread(master_path)
            master_img = master_img.astype('float32')
            master_img = (master_img - min_range_master) * 1.0 / (max_range_master - min_range_master)
            master_img -= float(mean_master_str)
            master_img = np.expand_dims(master_img, axis=-1)
            master_img = np.expand_dims(master_img, axis=0)  # add dimension for timestamp
            train_master.append(master_img)

            # read slave corresponding to pose
            min_range_slave_1 = float(range_slave_1_str.split(',')[0])
            max_range_slave_1 = float(range_slave_1_str.split(',')[1])
            slave_1_path = slave_1_dir + '/' + sampled_files[k].split(',')[1]  # idx 1 is always for the slave!
            # normalize slave image
            slave_1_img = misc.imread(slave_1_path, mode='RGB')
            slave_1_img = slave_1_img.astype('float32')
            slave_1_img[:, :, [0, 1, 2]] = slave_1_img[:, :, [2, 1, 0]]
            slave_1_img = (slave_1_img - min_range_slave_1) * 1.0 / (max_range_slave_1 - min_range_slave_1)
            # slave_1_img -= float(mean_master_str)
            slave_1_img[:, :, 0] -= float(mean_slave_1_str.split(",")[0])
            slave_1_img[:, :, 1] -= float(mean_slave_1_str.split(",")[1])
            slave_1_img[:, :, 2] -= float(mean_slave_1_str.split(",")[2])
            slave_1_img = np.expand_dims(slave_1_img, axis=0)  # add dimension for timestamp
            train_slave_1.append(slave_1_img)

            # read IMU data
            # the imu data starts at column 10 in sampled_files for 1 slave
            # the imu data starts at column 11 in sampled_files for 2 slaves
            imu_start = 11
            for l in range(20):
                # notes that we have loaded imu data in 1x120 format, and we need to save it in 20x6
                # rstrip() -> remove trailing new line \n
                train_slave_2[k][l] = np.array(sampled_files[k].rstrip().split(',')[imu_start:(imu_start + 6)],
                                                dtype=np.float64)
                imu_start += 6
            total_img_counter += 1
            print('Processing folder: ', all_exps[j], 'Total img idx ', str(total_img_counter),
                  ': ', sampled_files[k].split(',')[0], '. Master size: ', np.shape(train_master),
                  '. Slave 1 size: ', np.shape(train_slave_1),
                  '. Slave 2 size: ', np.shape(train_slave_2))

        print('Saving to h5 file ....')
        train_timestamp_np = np.array(train_timestamp)
        train_master_data_np = np.array(train_master)
        train_master_data_np = np.expand_dims(train_master_data_np, axis=0)  # add dimension for batch
        train_slave_1_data_np = np.array(train_slave_1)
        train_slave_1_data_np = np.expand_dims(train_slave_1_data_np, axis=0)  # add dimension for batch
        train_slave_2_data_np = np.array(train_slave_2)
        train_slave_2_data_np = np.expand_dims(train_slave_2_data_np, axis=0)  # add dimension for batch
        train_label_np = np.array(train_label)

        print('Data has been collected:')
        print('Master => ', master, ': ', np.shape(train_master_data_np))
        print('Slave 1 => ', slave_1, ': ', np.shape(train_slave_1_data_np))
        print('Slave 2 => ', slave_2, ': ', np.shape(train_slave_2_data_np))
        print('Label : ', np.shape(train_label_np))

        file_save = join(save_dir, 'turtle_seq_' + str(seq_counter) + '.h5')
        with h5py.File(file_save, 'w') as hf:
            hf.create_dataset('timestamp', data=np.array(train_timestamp_np).astype(int))
            hf.create_dataset(str(master + '_data'), data=train_master_data_np)
            hf.create_dataset(str(slave_1 + '_data'), data=train_slave_1_data_np)
            hf.create_dataset(str(slave_2 + '_data'), data=train_slave_2_data_np)
            hf.create_dataset('label_data', data=train_label_np)
        print('Finished! File saved in: ' + file_save)

        seq_counter += 1

    return 0

if __name__ == '__main__':
    main()