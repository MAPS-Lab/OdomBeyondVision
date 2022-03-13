#!/usr/bin/python
import datetime as dt
from datetime import datetime as dtime
import argparse
import os
import pandas

DESCRIPTION = """This script receives a path of directory to thermal images and match 
                with the pseudo ground truth csv according to their timestamp."""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--sensor_dir', required=True, help='''Specify the thermal
                    folder you want to associate with lidar timestamp.''')
parser.add_argument('--gt', required=True, help='''Specify the lidar pose
                    file (.csv) you want to match with the rgb timestamps.''')
parser.add_argument('--save_dir', help='''Specify
                    where to save the output file.''')

args = parser.parse_args()

image_files = [f for f in os.listdir(args.sensor_dir) if
               os.path.isfile(os.path.join(args.sensor_dir, f)) and
               f.endswith('png')]
image_files = [name + '\n' for name in image_files]

# load the gt into a df and then grep the useful 8 columns: timestamp, translation (x,y,z), rotation (quaternion)
df_gt = pandas.read_csv(args.gt)
gt = df_gt[['rosbagTimestamp', 'x', 'y', 'z', 'x.1', 'y.1', 'z.1', 'w']]
csv_path = os.path.join(args.save_dir, 'odom_' + os.path.splitext(os.path.basename(args.sensor_dir))[0].split('_')[0] + '_core.csv')
gt.to_csv(csv_path, sep=',', header=False, index=False)

with open(csv_path, 'r') as gt_file:
    gt_lines = [line for line in gt_file]

# converted_gt_lines = []
# for line in gt_lines:
#     new_time_gt = str(int(float(line.split(',')[0].partition("E")[0]) * 1e18))
#     new_string = new_time_gt + ',' + ','.join(line.split(',')[1:])
#     converted_gt_lines.append(new_string)

log_strings = image_files + gt_lines

log_strings.sort()

log_string_filename = os.path.join(args.save_dir, 'odom_' + os.path.splitext(os.path.basename(args.sensor_dir))[0].split('_')[0] + '_logs.txt')

with open(log_string_filename, 'w') as fw:
    for pos, line in enumerate(log_strings):
        fw.write(line)

filename = os.path.join(args.save_dir, 'odom_' + os.path.splitext(os.path.basename(args.sensor_dir))[0].split('_')[0] + '_ref.csv')
MAX_MINUTES = 600
img_counter = 0
with open(filename, 'w') as fw:
    for pos, line in enumerate(log_strings):
        if line[:-1].endswith('png'): #Just enough not to be confused with LIDAR poses

            # for thermal
            current_utc_time = dtime.utcfromtimestamp(int(line.split('.')[0]) / 1e9).strftime('%Y-%m-%d-%H-%M-%S-%f')
            current_datetime = dt.datetime.strptime(current_utc_time,"%Y-%m-%d-%H-%M-%S-%f")

            dists = [dt.timedelta(minutes=MAX_MINUTES), dt.timedelta(minutes=MAX_MINUTES)]
            if pos < len(log_strings)-1: #Do I have a next line still?
                #Check if next line is also an image
                if not log_strings[pos+1][:-1].endswith('png'):
                    next_utc_time = dtime.utcfromtimestamp(int(log_strings[pos+1].split(',')[0]) / 1e9).strftime('%Y-%m-%d-%H-%M-%S-%f')
                    next_datetime = dt.datetime.strptime(next_utc_time,"%Y-%m-%d-%H-%M-%S-%f")
                    dists[1] = abs(next_datetime - current_datetime)
                    print(next_utc_time)
            if pos > 0: #I have a previous line
                if not log_strings[pos-1][:-1].endswith('png'):
                    previous_utc_time = dtime.utcfromtimestamp(int(log_strings[pos-1].split(',')[0]) / 1e9).strftime('%Y-%m-%d-%H-%M-%S-%f')
                    previous_datetime = dt.datetime.strptime(previous_utc_time,"%Y-%m-%d-%H-%M-%S-%f")
                    dists[0] = abs(previous_datetime - current_datetime)
            shortest_dist = min(dists)
            # print(shortest_dist)

            if shortest_dist < dt.timedelta(minutes=MAX_MINUTES):
                shortest_idx = dists.index(min(dists))
                write_line = line[:-1] + ',' + log_strings[pos + 2*shortest_idx -1][:-1] + ','+  str(shortest_dist) + '\n'
                print(write_line)
                fw.write(write_line)

            img_counter += 1