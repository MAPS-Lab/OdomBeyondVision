import os
import yaml
from os.path import join, dirname
import inspect
import json

DESCRIPTION = """This script automatically generates datasets from massive data collections, e.g. 6 sensors."""

# Please use config.yaml to custom your dataset creation
parent_dir = dirname(dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
with open(join(parent_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

all_exp_files = cfg['dataset_creation_robot']['all_exp_files']
total_training_files = cfg['dataset_creation_robot']['test_file_idx']
train_files = all_exp_files[0:total_training_files]  # make sure you sort your sequence into: list training seq, list testing seq
dataroot = cfg['dataset_creation_robot']['dataroot']
master = cfg['dataset_creation_robot']['master']
slaves = cfg['dataset_creation_robot']['slaves']
master_gap = cfg['dataset_creation_robot']['master_gap']
saved_dir_h5 = cfg['dataset_creation_robot']['saved_dir_h5'] + '_gap' + str(master_gap)

if not os.path.exists(saved_dir_h5):
    os.makedirs(saved_dir_h5)

# save the seq.
dict = {}
for key, val in enumerate(all_exp_files):
    dict[key + 1] = val
with open(join(saved_dir_h5, 'seq_idx.json'), 'w') as fp:
    json.dump(dict, fp)

######################
# Generate sensor data association
######################

# for exp_date in all_exp_files:
#     current_dir = join(dataroot, exp_date)
#     print('Processing directory: ', current_dir)
#
#     # Associate master with GT
#     sensor_dir = join(current_dir, master)
#     gt_file = join(current_dir, '_slash_odom.csv')
#     cmd = 'python -W ignore ' + 'associate_sensor_to_gt.py' + ' --sensor_dir ' + sensor_dir + \
#           ' --gt ' + gt_file + ' --save_dir ' + current_dir
#     print(cmd)
#     os.system(cmd)
#
#     # Associate master with 1st slave
#     slave_dir = join(current_dir, slaves[0])
#     sync_file = join(current_dir, str('odom_' + master + '_ref.csv'))
#     cmd = 'python -W ignore ' + 'sync_ref_sensor_to_slave.py' + ' --slave_dir ' + slave_dir + ' --sync_file ' \
#           + sync_file + ' --column_idx ' + str(1) + ' --save_dir ' + ' ' + current_dir
#     print(cmd)
#     os.system(cmd)
#
#     # Associate master with 2nd slave
#     slave_dir = join(current_dir, slaves[1])
#     sync_file = join(current_dir, str('odom_' + master + '_ref_' + slaves[0] + '.csv'))
#     cmd = 'python -W ignore ' + 'sync_ref_sensor_to_slave.py' + ' --slave_dir ' + slave_dir + ' --sync_file ' \
#           + sync_file + ' --column_idx ' + str(2) + ' --save_dir ' + ' ' + current_dir
#     print(cmd)
#     os.system(cmd)
#
#     # Associate master with 3rd slave
#     slave_dir = join(current_dir, slaves[2])
#     sync_file = join(current_dir, str('odom_' + master + '_ref_' + slaves[0] + '_' + slaves[1] + '.csv'))
#     cmd = 'python -W ignore ' + 'sync_ref_sensor_to_slave.py' + ' --slave_dir ' + slave_dir + ' --sync_file ' \
#           + sync_file + ' --column_idx ' + str(3) + ' --save_dir ' + ' ' + current_dir
#     print(cmd)
#     os.system(cmd)
#
#     # Associate master with 4th slave
#     slave_dir = join(current_dir, slaves[3])
#     sync_file = join(current_dir, str('odom_' + master + '_ref_' + slaves[0] + '_' + slaves[1] + '_' + slaves[2] + '.csv'))
#     cmd = 'python -W ignore ' + 'sync_ref_sensor_to_slave.py' + ' --slave_dir ' + slave_dir + ' --sync_file ' \
#           + sync_file + ' --column_idx ' + str(4) + ' --save_dir ' + ' ' + current_dir
#     print(cmd)
#     os.system(cmd)
#
#     # Associate master with 5th slave, which is IMU
#     slave_file = join(current_dir, '_slash_imu_slash_data.csv')  # for slave imu file
#     sync_file = join(current_dir, str('odom_' + master + '_ref_' + slaves[0] + '_' + slaves[1] + '_' + slaves[2] + '_' + slaves[3] + '.csv'))
#     cmd = 'python -W ignore ' + 'sync_ref_sensor_to_slave_imu.py' + ' --slave_imu ' + slave_file + ' --sync_file ' \
#           + sync_file + ' --save_dir ' + ' ' + current_dir
#     print(cmd)
#     os.system(cmd)


# #######################
# # Compute range and dataset mean
# #######################

ref_file_name = str('odom_' + master + '_ref_' + slaves[0] + '_' + slaves[1] + '_' + slaves[2] + '_' + slaves[3] + '_' + slaves[4] + '.csv')

# # Check whether the master requires custom range value (either thermal/depth). If yes then we need to compute range.
# is_need_thermal_range = 0
# is_need_depth_range = 0
# thermal_idx = 0
# depth_idx = 0
# if master == 'thermal':
#     is_need_thermal_range = 1
#     thermal_idx = 0
# elif master == 'depth':
#     is_need_depth_range = 1
#     depth_idx = 0
# elif master != 'thermal' and master != 'depth':
#     file_range = open(join(saved_dir_h5, str('dataset_range_' + master + '.txt')), "w")
#     L = [str(0), ',', str(255)]
#     file_range.writelines(L)
#     file_range.close()
#
# for i in range(len(slaves)):
#     if slaves[i] == 'thermal':
#         is_need_thermal_range = 1
#         thermal_idx = i + 1
#     elif slaves[i] == 'depth':
#         is_need_depth_range = 1
#         depth_idx = i + 1
#
# print(is_need_thermal_range)
# print(is_need_depth_range)
#
# # compute range value of the thermal and depth data
# if is_need_thermal_range == 1:
#     cmd = 'python -W ignore ' + 'compute_dataset_range_value.py' + ' --dataroot ' + dataroot + ' --saved_dir_h5 ' + saved_dir_h5 \
#           + ' --data_type ' + 'thermal' + ' --ref_file_name ' + ref_file_name + ' --ref_col_idx ' + str(thermal_idx) + \
#           ' --gap ' + str(master_gap)
#     print(cmd)
#     os.system(cmd)
#
#
# if is_need_depth_range == 1:
#     cmd = 'python -W ignore ' + 'compute_dataset_range_value.py' + ' --dataroot ' + dataroot + ' --saved_dir_h5 ' + saved_dir_h5 \
#           + ' --data_type ' + 'depth' + ' --ref_file_name ' + ref_file_name + ' --ref_col_idx ' + str(depth_idx) + \
#           ' --gap ' + str(master_gap)
#     print(cmd)
#     os.system(cmd)
#
# # compute dataset mean the master
# range_master = join(saved_dir_h5, str('dataset_range_' + master + '.txt'))
# cmd = 'python -W ignore ' + 'compute_dataset_mean.py' + ' --dataroot ' + dataroot + ' --saved_dir_h5 ' + saved_dir_h5 \
#       + ' --data_type ' + master + ' --ref_file_name ' + ref_file_name + ' --ref_col_idx ' + str(0) \
#       + ' --range_file ' + range_master + ' --gap ' + str(master_gap)
# print(cmd)
# os.system(cmd)
#
# # compute dataset mean for the slaves
# for i in range(len(slaves) - 1):  # minus IMU
#     if slaves[i] != 'thermal' and slaves[i] != 'depth':
#         file_range = open(join(saved_dir_h5, str('dataset_range_' + slaves[i] + '.txt')), "w")
#         L = [str(0), ',', str(255)]
#         file_range.writelines(L)
#         file_range.close()
#
#     range_slave = join(saved_dir_h5, str('dataset_range_' + slaves[i] + '.txt'))
#     cmd = 'python -W ignore ' + 'compute_dataset_mean.py' + ' --dataroot ' + dataroot + ' --saved_dir_h5 ' + saved_dir_h5  \
#           + ' --data_type ' + slaves[i] + ' --ref_file_name ' + ref_file_name + ' --ref_col_idx ' + str(i + 1) \
#           + ' --range_file ' + range_slave + ' --gap ' + str(master_gap)
#     print(cmd)
#     os.system(cmd)

#######################
# generate h5 files
#######################

cmd = 'python -W ignore ' + 'create_h5_dataset.py' + ' ' + ' --dataroot ' + dataroot + ' --ref_file_name ' + ref_file_name + \
    ' --save_dir ' + saved_dir_h5 + ' --gap ' + str(master_gap)
print(cmd)
os.system(cmd)
