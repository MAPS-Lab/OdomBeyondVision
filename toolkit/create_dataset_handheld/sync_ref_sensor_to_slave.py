#!/usr/bin/python
import argparse
import os

DESCRIPTION = """This script receives a synchronized file from a ref sensor, and match 
                with the another slave sensor according to their timestamp."""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--slave_dir', required=True, help='''Specify the slave sensor
                    folder you want to associate with the sync file.''')
parser.add_argument('--sync_file', required=True, help='''Specify the synchronized
                    file (.csv) you want to match with the slave's timestamps.''')
parser.add_argument('--column_idx', required=True, help='''Specity the index to which column in csv file we want to 
                                                    insert the new data.''')
parser.add_argument('--save_dir', help='''Specify where to save the output file.''')

args = parser.parse_args()

col_idx = int(args.column_idx)

image_files = [f for f in os.listdir(args.slave_dir) if
               os.path.isfile(os.path.join(args.slave_dir, f)) and
               f.endswith('png')]
image_files = [name + '\n' for name in image_files] # cut the word thermalIR_

with open(args.sync_file, 'r') as sync_files:
    sync_lines = [line for line in sync_files]

log_strings = image_files + sync_lines

log_strings.sort()

filename = os.path.join(args.save_dir, os.path.splitext(os.path.basename(args.sync_file))[0] + '_' +
                        os.path.splitext(os.path.basename(args.slave_dir))[0] + '.csv')
MAX_MINUTES = 600
img_counter = 0
with open(filename, 'w') as fw:
    for pos, line in enumerate(log_strings):
        if not line[:-1].endswith('png'): #Just enough not to be confused with mmwave images
            for i in range(1, 100):
                if log_strings[pos-i][:-1].endswith('png'):
                    if col_idx == 1:
                        write_line = line[:-1].split(',')[0] + ',' + log_strings[pos-i][:-1] + ',' + ','.join(
                            line.split(',')[1:])
                    if col_idx > 1:
                        write_line = ','.join(line[:-1].split(',')[0:col_idx]) + ',' + log_strings[pos - i][:-1] + ',' + ','.join(
                            line.split(',')[col_idx:])
                    print(write_line)
                    fw.write(write_line)
                    break

            img_counter += 1
