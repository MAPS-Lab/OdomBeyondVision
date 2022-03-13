#!/usr/bin/python
import argparse
import os
import pandas

DESCRIPTION = """This script receives a synchronized file from a ref sensor, and match
                with the another slave sensor (IMU) according to their timestamp."""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--slave_imu', required=True, help='''Specify the slave sensor
                    file you want to associate with the sync file.''')
parser.add_argument('--sync_file', required=True, help='''Specify the synchronized
                    file (.csv) you want to match with the slave's timestamps.''')
parser.add_argument('--save_dir', help='''Specify
                    where to save the output file.''')

args = parser.parse_args()

# read imu file
df_imu = pandas.read_csv(args.slave_imu)
imu_core = df_imu[['rosbagTimestamp', 'seq', 'x.2', 'y.2', 'z.2', 'x.1', 'y.1', 'z.1']]
csv_path = os.path.join(args.save_dir, 'odom_' + str(os.path.splitext(os.path.basename(args.slave_imu))[0].split('_')[2]) + '_core.csv')
imu_core.to_csv(csv_path, sep=',', header=False, index=False)
with open(csv_path, 'r') as imu_file:
    imu_lines = [line for line in imu_file]

# Read sync file from last step
with open(args.sync_file, 'r') as sync_files:
    sync_lines = [line for line in sync_files]

log_strings = imu_lines + sync_lines
log_strings.sort()
log_string_filename = os.path.join(args.save_dir, 'odom_' + str(os.path.splitext(os.path.basename(args.slave_imu))[0].split('_')[2]) + '_logs.txt')

with open(log_string_filename, 'w') as fw:
    for pos, line in enumerate(log_strings):
        fw.write(line)

# Write a new sync file for next step
filename = os.path.join(args.save_dir, os.path.splitext(os.path.basename(args.sync_file))[0] + '_' +
                        str(os.path.splitext(os.path.basename(args.slave_imu))[0].split('_')[2]) + '.csv')
imu_iter = 0
with open(filename, 'w') as fw:
    for pos, line in enumerate(log_strings):
        if pos > 40:  # to get enough imu
            if line.split(',')[0].endswith('png'):
                new_line = line[:-1]
                # current_imu = []
                for i in range(50):
                    if not log_strings[pos - 1 - i].split(',')[0].endswith('png'):
                        # Got your imu data, collect it 30 data maximum (note that in logs string, there are .png file as well
                        # so we iterate until 50. If we got 30 imu already, then we break.
                        new_line += ',' + ','.join(log_strings[pos - 1 - i][:-1].split(',')[2:])
                        # current_imu.extend(log_strings[pos-1-i])
                        imu_iter += 1
                        if imu_iter == 30:  # we save 30 imu values, back in time
                            imu_iter = 0
                            new_line += '\n'
                            break
                fw.write(new_line)
