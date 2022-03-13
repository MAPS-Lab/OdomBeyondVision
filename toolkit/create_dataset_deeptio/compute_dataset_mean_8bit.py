from scipy.misc import imread
import numpy as np
import argparse
import os
from os.path import join, dirname
import inspect
import yaml
import cv2

DESCRIPTION = """This script computes a dataset mean for particular modality."""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--dataroot', required=True, help='''Specify the dataroot directory.''')
parser.add_argument('--data_type', required=True, help='''Specify the data type, e.g. thermal, depth, etc.''')
parser.add_argument('--ref_file_name', required=True, help='''Specify the reference (synchronized) filename to load the data''')
parser.add_argument('--ref_col_idx', required=True, help='''Specify the column index in the ref file to obtain the list images''')
parser.add_argument('--range_file', required=True, help='''Specify the range file.''')
parser.add_argument('--gap', required=True, help='''Specify the sampling gap.''')
args = parser.parse_args()

dataroot = args.dataroot
data_type = args.data_type
ref_file = args.ref_file_name
ref_col_idx = int(args.ref_col_idx)
range_file = args.range_file
GAP = int(args.gap)

parent_dir = dirname(dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
with open(join(parent_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

train_files = cfg['dataset_creation']['train_files']

def get_1channel_image(img_path, min_range, max_range):
    img = imread(img_path)
    img = img.astype('float32')
    # np.clip(img, 0, 1, out=img)
    img = cv2.normalize(img, None, min_range, max_range, cv2.NORM_MINMAX)
    img = (img - min_range) * 1.0 / (max_range - min_range)
    return img


def get_3channel_image(img_path, min_range, max_range):
    img = imread(img_path, mode='RGB')
    img = img.astype('float32')
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]] # swap channels
    # np.clip(img, 0, 1, out=img)
    img = cv2.normalize(img, None, min_range, max_range, cv2.NORM_MINMAX)
    img = (img - min_range) * 1.0 / (max_range - min_range)
    return img


sum_R = 0
sum_G = 0
sum_B = 0
num_images = 0

file_range = open(range_file, "r")
data_range_str = file_range.readlines()[0]
file_range.close()

for train_file in train_files:
    img_dir = join(dataroot, train_file, data_type)
    file_full_path = join(dataroot, train_file, ref_file)

    with open(file_full_path, 'r') as the_files:
        file_lines = [line for line in the_files]
    sampled_files = []
    for k in range(0, np.size(file_lines), GAP):
        sampled_files.append(file_lines[k])
    for k in range(len(sampled_files)):
        img_path = join(img_dir, sampled_files[k].split(',')[ref_col_idx])
        if data_type == 'thermal' or data_type == 'mmwave' or data_type == 'lidar':
            min_range = float(data_range_str.split(',')[0])
            max_range = float(data_range_str.split(',')[1])
            img = get_1channel_image(img_path, min_range, max_range)
            sum_R += np.sum(img)
        elif data_type == 'rgb':
            min_range = float(data_range_str.split(',')[0])
            max_range = float(data_range_str.split(',')[1])
            img = get_3channel_image(img_path, min_range, max_range)
            sum_R += np.sum(img[:, :, 0])
            sum_G += np.sum(img[:, :, 1])
            sum_B += np.sum(img[:, :, 2])

        print('Load img ', str(num_images), '. File: ', img_path, 'Sum R value: ', str(sum_R))
        num_images += 1

width = img.shape[1]
height = img.shape[0]

total_pixels = num_images*width*height

print('mean')
if data_type == 'thermal' or data_type == 'mmwave' or data_type == 'lidar':
    mean_r = sum_R / (total_pixels)
    print(mean_r)
    file1 = open(join(dataroot, str("dataset_mean_" + data_type + ".txt")), "w")
    L = [str(mean_r)]
    file1.writelines(L)
    file1.close()
elif data_type == 'rgb':
    mean_r = sum_R / (total_pixels)
    print(mean_r)
    mean_g = sum_G / (total_pixels)
    print(mean_g)
    mean_b = sum_B / (total_pixels)
    print(mean_b)
    file1 = open(join(dataroot, str("dataset_mean_" + data_type + ".txt")), "w")
    L = [str(mean_r), ',' , str(mean_g), ',' , str(mean_b)]
    file1.writelines(L)
    file1.close()