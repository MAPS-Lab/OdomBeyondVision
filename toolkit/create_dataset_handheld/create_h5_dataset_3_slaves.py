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


def load_normalize_1channel_img(range_str, mean_str, data_dir, img_name):
    min_range = float(range_str.split(',')[0])
    max_range = float(range_str.split(',')[1])
    # master_path = data_dir + '/' + sampled_files[k].split(',')[0]  # idx 0 is always for the master!
    img_path = data_dir + '/' + img_name
    # normalize master image
    img = misc.imread(img_path)
    img = img.astype('float32')
    img = (img - min_range) * 1.0 / (max_range - min_range)
    img -= float(mean_str)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)  # add dimension for timestamp
    return img


def load_normalize_3channel_img(range_str, mean_str, data_dir, img_name):
    min_range = float(range_str.split(',')[0])
    max_range = float(range_str.split(',')[1])
    # master_path = data_dir + '/' + sampled_files[k].split(',')[0]  # idx 0 is always for the master!
    img_path = data_dir + '/' + img_name
    # normalize master image
    img = misc.imread(img_path, mode='RGB')
    img = img.astype('float32')
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = (img - min_range) * 1.0 / (max_range - min_range)
    img[:, :, 0] -= float(mean_str.split(",")[0])
    img[:, :, 1] -= float(mean_str.split(",")[1])
    img[:, :, 2] -= float(mean_str.split(",")[2])
    img = np.expand_dims(img, axis=0)  # add dimension for timestamp
    return img

def main():
    print('FOR **Master and 2 slaves** ONLY!')

    DESCRIPTION = """This script receives a working directory and a dataset mean for each modality."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataroot', required=True, help='''Specify the dataroot directory.''')
    parser.add_argument('--ref_file_name', required=True,
                        help='''Specify the reference (synchronized) filename to load the data''')
    parser.add_argument('--save_dir', help='''Specify save directory.''')
    parser.add_argument('--gap', required=True, help='''Specify the sampling gap.''')

    args = parser.parse_args()

    dataroot = args.dataroot
    ref_file_name = args.ref_file_name
    GAP = int(args.gap)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load all sequences name and list of master-slaves
    parent_dir = dirname(dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    with open(join(parent_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    all_exps = cfg['dataset_creation_handheld']['all_exp_files']
    master = cfg['dataset_creation_handheld']['master']
    slaves = cfg['dataset_creation_handheld']['slaves']

    # Prepare the path files to load dataset range and mean
    range_master_file = join(save_dir, str('dataset_range_' + master + '.txt'))
    range_slave_files = []
    for i in range(len(slaves) - 1): # minus IMU
        range_slave_files.append(join(save_dir, str('dataset_range_' + slaves[i] + '.txt')))

    mean_master_file = join(save_dir, str('dataset_mean_' + master + '.txt'))
    mean_slave_files = []
    for i in range(len(slaves) - 1):
        mean_slave_files.append(join(save_dir, str('dataset_mean_' + slaves[i] + '.txt')))

    #  === Load dataset range and mean ===
    #  Master
    file_range_master = open(range_master_file, "r")
    range_master_str = file_range_master.readlines()[0]
    file_range_master.close()

    file_mean_master = open(mean_master_file, "r")
    mean_master_str = file_mean_master.readlines()[0]
    file_mean_master.close()

    range_slave_str = []
    mean_slave_str = []
    for i in range(len(slaves) - 1):
        file_range_slave = open(range_slave_files[i], "r")
        range_slave_str.append(file_range_slave.readlines()[0])
        file_range_slave.close()

        file_mean_slave = open(mean_slave_files[i], "r")
        mean_slave_str.append(file_mean_slave.readlines()[0])
        file_mean_slave.close()

    # Specify the gap for all sequences
    odom_data_GAP = [GAP] * len(all_exps)
    seq_counter = 1
    total_img_counter = 0

    for j in range(len(all_exps)):
        # Get reference/association files
        file_full_path = join(dataroot, all_exps[j], ref_file_name)
        with open(file_full_path, 'r') as the_files:
            file_lines = [line for line in the_files]

        # Sample file based on the specified gap
        sampled_files = []
        sampling = odom_data_GAP[j]
        for k in range(0, np.size(file_lines), sampling):
            sampled_files.append(file_lines[k])

        # Variables to save data
        train_timestamp = []
        train_master = []
        train_slave_1 = []
        train_slave_2 = []
        train_slave_imu = np.empty((len(sampled_files), 30, 6), dtype=np.float64) # imu
        train_label = []

        # save timestamp
        timestamp = [line[:-1].split(',')[3] for line in sampled_files] # timestamp idx for new ref is 5
        print('Total timestamp: ', np.shape(timestamp))
        train_timestamp.append(timestamp)

        gt_lines_float = []
        for line in sampled_files:
            gt_lines_float.append(np.array(
                [float(line[:-1].split(',')[3]),  # timestamp
                 float(line[:-1].split(',')[4]), float(line[:-1].split(',')[5]), float(line[:-1].split(',')[6]),
                 # translation
                 float(line[:-1].split(',')[7]), float(line[:-1].split(',')[8]),
                 float(line[:-1].split(',')[9]), float(line[:-1].split(',')[10])]))  # quaternion

        lidar_rel_poses = rotated_to_local(gt_lines_float)
        train_label.append(lidar_rel_poses)
        print('GT size: ', np.shape(train_label))

        for k in range(0, len(sampled_files)):

            # Load & normalize master img
            master_dir = join(dataroot, all_exps[j], master)
            if master == 'thermal' or master == 'mmwave_middle' or master == 'lidar' or master == 'depth':
                master_img = load_normalize_1channel_img(range_master_str, mean_master_str, master_dir,
                                                         sampled_files[k].split(',')[0])
            elif master == 'rgb':
                master_img = load_normalize_3channel_img(range_master_str, mean_master_str, master_dir,
                                                         sampled_files[k].split(',')[0])
            train_master.append(master_img)

            # Load & normalize slave 1-4
            for i in range(len(slaves)-1): # this 'i' is only used in this loop
                slave_dir = join(dataroot, all_exps[j], slaves[i])
                if slaves[i] == 'thermal' or slaves[i] == 'mmwave_middle' or slaves[i] == 'lidar' or slaves[i] == 'depth':
                    slave_img = load_normalize_1channel_img(range_slave_str[i], mean_slave_str[i], slave_dir,
                                                      sampled_files[k].split(',')[i+1])
                elif slaves[i] == 'rgb':
                    slave_img = load_normalize_3channel_img(range_slave_str[i], mean_slave_str[i], slave_dir,
                                                            sampled_files[k].split(',')[i + 1])
                if i == 0:
                    train_slave_1.append(slave_img)
                elif i == 1:
                    train_slave_2.append(slave_img)

            # Load IMU data
            imu_start = 12 # 14 - (5 - 3) # 4 slaves out of 5
            for l in range(30):
                # notes that we have loaded imu data in 1x180 format, and we need to save it in 30x6
                # rstrip() -> remove trailing new line \n
                train_slave_imu[k][l] = np.array(sampled_files[k].rstrip().split(',')[imu_start:(imu_start + 6)],
                                                dtype=np.float64)
                imu_start += 6
            total_img_counter += 1
            print('Processing folder: ', all_exps[j], 'Total img idx ', str(total_img_counter),
                  '. Master ', master , ' size: ', np.shape(train_master),
                  '. Slave ', slaves[0] ,' size: ', np.shape(train_slave_1),
                  '. Slave ', slaves[1] ,' size: ', np.shape(train_slave_2),
                  '. Slave ', slaves[2] ,' size: ', np.shape(train_slave_imu),)

        print('Saving to h5 file ....')
        train_timestamp_np = np.array(train_timestamp)
        train_master_data_np = np.array(train_master)
        train_slave_1_data_np = np.array(train_slave_1)
        train_slave_2_data_np = np.array(train_slave_2)
        train_slave_imu_data_np = np.array(train_slave_imu)
        train_label_np = np.array(train_label)

        data_summary_path = join(save_dir, 'dataset_summary.txt')
        with open(data_summary_path, 'a+') as f:
            print('Data descripton for sequence: ', all_exps[j], file=f)
            print('Master => ', master, ': ', np.shape(train_master_data_np), file=f)
            print('Slave 1 => ', slaves[0], ': ', np.shape(train_slave_1_data_np), file=f)
            print('Slave 2 => ', slaves[1], ': ', np.shape(train_slave_2_data_np), file=f)
            print('Slave 5 => ', slaves[2], ': ', np.shape(train_slave_imu_data_np), file=f)
            print('Label : ', np.shape(train_label_np), file=f)

        file_save = join(save_dir, 'turtle_seq_' + str(seq_counter) + '.h5')
        with h5py.File(file_save, 'w') as hf:
            hf.create_dataset('timestamp', data=np.array(train_timestamp_np).astype(int))
            hf.create_dataset(str(master + '_data'), data=train_master_data_np)
            hf.create_dataset(str(slaves[0] + '_data'), data=train_slave_1_data_np)
            hf.create_dataset(str(slaves[1] + '_data'), data=train_slave_2_data_np)
            hf.create_dataset(str(slaves[2] + '_data'), data=train_slave_imu_data_np)
            hf.create_dataset('label_data', data=train_label_np)
        print('Finished! File saved in: ' + file_save)
        seq_counter += 1

    return 0

if __name__ == '__main__':
    main()