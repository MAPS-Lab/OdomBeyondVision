from scipy.misc import imread
import numpy as np
import argparse
import os
from os.path import join, dirname
import inspect
import yaml

DESCRIPTION = """This script receives a working directory and a list of training exps,
                    and outputs a range values from the whole dataset.
                    This is usually used for computing thermal range data."""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--dataroot', required=True, help='''Specify the dataroot directory.''')
parser.add_argument('--saved_dir_h5', required=True, help='''Specify where mean files will be saved.''')
parser.add_argument('--data_type', required=True, help='''Specify the data type, e.g. thermal, depth, etc.''')
parser.add_argument('--ref_file_name', required=True, help='''Specify the reference (synchronized) filename to load the data''')
parser.add_argument('--ref_col_idx', required=True, help='Specify which column to look at in ref file to load the images.')
parser.add_argument('--gap', required=True, help='''Specify the sampling gap.''')
args = parser.parse_args()

dataroot = args.dataroot
saved_dir_h5 = args.saved_dir_h5
data_type = args.data_type
ref_file_name = args.ref_file_name
data_col_idx = int(args.ref_col_idx)
GAP = int(args.gap)

parent_dir = dirname(dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
with open(join(parent_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

all_exp_files = cfg['dataset_creation_robot']['all_exp_files']
total_training_files = cfg['dataset_creation_robot']['test_file_idx']
# train_files = all_exp_files[0:total_training_files]
train_files = all_exp_files

num_images = 0
for j in range(len(train_files)):
    img_dir = join(dataroot, train_files[j], data_type)
    file_full_path = join(dataroot, train_files[j], ref_file_name)
    with open(file_full_path, 'r') as the_files:
        file_lines = [line for line in the_files]
    sampled_files = []
    for k in range(0, np.size(file_lines), GAP):
        sampled_files.append(file_lines[k])
    for k in range(len(sampled_files)):
        # we always assume that the first column in ref_file is the one we used!!!
        img_path = img_dir + '/' + sampled_files[k].split(',')[data_col_idx]
        num_images += 1
        img = imread(img_path)
        img = img.astype('float32')
        img = np.array(img)
        img = img.flatten()
        print('Idx j :', str(j), 'Idx k :', str(k), ' Idx all: ', str(num_images),
              ' Load img : ' + img_path)
        if (j == 0) and (k == 0):
            global_min = np.min(img)
            global_max = np.max(img)
            print('Update global min to : ' + str(global_min))
            print('Update global max to : ' + str(global_max))
        temp_min = np.min(img)
        temp_max = np.max(img)
        if temp_min < global_min:
            global_min = temp_min
            print('Update global min to : ' + str(global_min))
        if temp_max > global_max:
            global_max = temp_max
            print('Update global max to : ' + str(global_max))

print('Global min : ' + str(global_min))
print('Global max : ' + str(global_max))
print('Range : ' + str(global_max - global_min))

file1 = open(join(saved_dir_h5, str('dataset_range_' + data_type + '.txt')), "w")
L = [str(global_min), ',', str(global_max)]
file1.writelines(L)
file1.close()
