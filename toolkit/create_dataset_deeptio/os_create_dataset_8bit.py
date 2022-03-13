import os
import yaml
from os.path import join, dirname
import inspect

DESCRIPTION = """This script automatically generates datasets from massive data collections (could be 1 or 2 slaves)."""

# Please use config.yaml to custom your dataset creation
parent_dir = dirname(dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
with open(join(parent_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

all_exp_files = cfg['dataset_creation']['all_exp_files']
train_files = cfg['dataset_creation']['train_files']
dataroot = cfg['dataset_creation']['dataroot']
is_2_slaves = cfg['dataset_creation']['is_2_slaves']
master = cfg['dataset_creation']['master']
slave_1 = cfg['dataset_creation']['slave_1']
slave_2 = cfg['dataset_creation']['slave_2']
master_gap = cfg['dataset_creation']['master_gap']
saved_dir_h5 = cfg['dataset_creation']['saved_dir_h5']

######################
# generate data association
######################

# Associate master
# for exp_date in all_exp_files:
#     current_dir = join(dataroot, exp_date)
#     print(current_dir)
#     sensor_dir = join(current_dir, master) # for RGB
#     gt_file = join(current_dir, '_slash_odom.csv')
#
#     cmd = 'python -W ignore ' + 'associate_sensor_to_gt.py' + ' --sensor_dir ' + sensor_dir + \
#           ' --gt ' + gt_file + ' --save_dir ' + current_dir
#
#     print(cmd)
#     os.system(cmd)
#
# # Associate 1st slave
# for exp_date in all_exp_files:
#     if slave_1 == 'imu':
#         current_dir = join(dataroot, exp_date)
#         slave_file = join(current_dir, '_slash_imu_slash_data.csv')  # for slave imu file
#         sync_file = join(current_dir, str('odom_' + master + '_ref.csv'))
#
#         cmd = 'python -W ignore ' + 'sync_ref_sensor_to_slave_imu.py' + ' --slave_imu ' + slave_file + ' --sync_file ' \
#               + sync_file + ' --save_dir ' + ' ' + current_dir
#         print(cmd)
#         os.system(cmd)
#     elif slave_1 != 'imu':
#         current_dir = join(dataroot, exp_date)
#         slave_dir = join(current_dir, slave_1)
#         sync_file = join(current_dir, str('odom_' + master + '_ref.csv'))
#         cmd = 'python -W ignore ' + 'sync_ref_sensor_to_slave.py' + ' --slave_dir ' + slave_dir + ' --sync_file '\
#               + sync_file + ' --save_dir ' + ' ' + current_dir
#         print(cmd)
#         os.system(cmd)
#
# # Associate 2nd slave
# if is_2_slaves == 1:
#     for exp_date in all_exp_files:
#         current_dir = join(dataroot, exp_date)
#         slave_file = join(current_dir, '_slash_imu_slash_data.csv')  # for slave imu file
#         sync_file = join(current_dir, str('odom_' + master + '_ref_' + slave_1 +'.csv'))
#         cmd = 'python -W ignore ' + 'sync_ref_sensor_to_slave_imu.py' + ' --slave_imu ' + slave_file + ' --sync_file ' \
#               + sync_file + ' --save_dir ' + ' ' + current_dir
#         print(cmd)
#         os.system(cmd)

#
# #######################
# # compute range and dataset mean
# #######################

# compute range value of the training data
if is_2_slaves == 1:
    ref_file_name = str('odom_' + master + '_ref_' + slave_1 + '_' + slave_2 + '.csv')
elif is_2_slaves == 0:
    ref_file_name = str('odom_' + master + '_ref_' + slave_1 + '.csv')
# #
if master == 'thermal':
    file1 = open(join(dataroot, str('dataset_range_' + master + '.txt')), "w")
    L = [str(0), ',', str(255)]
    file1.writelines(L)
    file1.close()
elif master != 'thermal':
    file1 = open(join(dataroot, str('dataset_range_' + master + '.txt')), "w")
    L = [str(0), ',' , str(255)]
    file1.writelines(L)
    file1.close()

if slave_1 != 'imu':
    file1 = open(join(dataroot, str('dataset_range_' + slave_1 + '.txt')), "w")
    L = [str(0), ',' , str(255)]
    file1.writelines(L)
    file1.close()

# compute dataset mean
col_idx_master = 0
col_idx_slave = 1
if is_2_slaves == 1:
    # mean for master
    range_master = join(dataroot, str('dataset_range_' + master + '.txt'))
    cmd = 'python -W ignore ' + 'compute_dataset_mean_8bit.py' + ' --dataroot ' + dataroot  \
          + ' --data_type ' + master + ' --ref_file_name ' + ref_file_name + ' --ref_col_idx ' + str(col_idx_master) \
          + ' --range_file ' + range_master + ' --gap ' + str(master_gap)
    print(cmd)
    os.system(cmd)
    # mean for 1st slave
    range_slave1 = join(dataroot, str('dataset_range_' + slave_1 + '.txt'))
    cmd = 'python -W ignore ' + 'compute_dataset_mean_8bit.py' + ' --dataroot ' + dataroot  \
          + ' --data_type ' + slave_1 + ' --ref_file_name ' + ref_file_name + ' --ref_col_idx ' + str(col_idx_slave) \
          + ' --range_file ' + range_slave1 + ' --gap ' + str(master_gap)
    print(cmd)
    os.system(cmd)
elif is_2_slaves == 0:
    range_master = join(dataroot, str('dataset_range_' + master + '.txt'))
    cmd = 'python -W ignore ' + 'compute_dataset_mean_8bit.py' + ' --dataroot ' + dataroot  \
          + ' --data_type ' + master + ' --ref_file_name ' + ref_file_name + ' --ref_col_idx ' + str(col_idx_master) \
          + ' --range_file ' + range_master + ' --gap ' + str(master_gap)
    print(cmd)
    os.system(cmd)


#######################
# generate h5 files
#######################

if is_2_slaves == 1:
    range_master_file = join(dataroot, str('dataset_range_' + master + '.txt'))
    range_slave1_file = join(dataroot, str('dataset_range_' + slave_1 + '.txt'))
    mean_master_file = join(dataroot, str('dataset_mean_' + master + '.txt'))
    mean_slave1_file = join(dataroot, str('dataset_mean_' + slave_1 + '.txt'))

    cmd = 'python -W ignore ' + 'create_h5_dataset_two_slaves_8bit.py' + ' ' + \
          ' --dataroot ' + dataroot + ' --ref_file_name ' + ref_file_name + \
          ' --master ' + master + ' --slave_1 ' + slave_1 + ' --slave_2 ' + slave_2 + \
          ' --mean_master_file ' + mean_master_file + ' --mean_slave1_file ' + mean_slave1_file + \
          ' --range_master_file ' + range_master_file + ' --range_slave1_file ' + range_slave1_file + \
          ' --save_dir ' + saved_dir_h5 + ' --gap ' + str(master_gap)
    print(cmd)
    os.system(cmd)

# todo is_2_slaves == 0