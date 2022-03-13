import struct
import csv
import numpy as np
import pandas as pd


def make_frames_from_csv(csv_path):
    readings_dict = dict()
    with open(csv_path, 'r') as input_file:
        reader = csv.reader(input_file)
        next(reader)
        for row in reader:
            pts = list()
            # add timestamp
            timestamp = row[0]  # timestamp = row[4] + row[5].zfill(9)
            # parsing
            try:
                offset_col = row[37]
            except:
                offset_col = row[29]

            pt_cloud = np.fromstring(offset_col[1:-1], dtype=int, sep=',')

            for i in range(0, int(len(pt_cloud) / 32)):
                point = list()
                # x
                tmp = struct.pack('4B', int(pt_cloud[32 * i]), int(pt_cloud[32 * i + 1]), int(pt_cloud[32 * i + 2]),
                                  int(pt_cloud[32 * i + 3]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # y
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 4]), int(pt_cloud[32 * i + 5]), int(pt_cloud[32 * i + 6]),
                                  int(pt_cloud[32 * i + 7]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # z
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 8]), int(pt_cloud[32 * i + 9]),
                                  int(pt_cloud[32 * i + 10]),
                                  int(pt_cloud[32 * i + 11]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # intensity
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 16]), int(pt_cloud[32 * i + 17]),
                                  int(pt_cloud[32 * i + 18]),
                                  int(pt_cloud[32 * i + 19]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # range
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 20]), int(pt_cloud[32 * i + 21]),
                                  int(pt_cloud[32 * i + 22]),
                                  int(pt_cloud[32 * i + 23]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # doppler
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 24]), int(pt_cloud[32 * i + 25]),
                                  int(pt_cloud[32 * i + 26]),
                                  int(pt_cloud[32 * i + 27]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                pts.append(point)
            readings_dict[timestamp] = pts

        return readings_dict


def make_frames_from_csv_doppler(csv_path):
    doppler_csv_path = csv_path.replace('.csv', '_scan.csv')
    print(doppler_csv_path)
    vel_data = pd.read_csv(doppler_csv_path)
    # print(vel_data.head())
    vel_idx = 0
    readings_dict = dict()
    with open(csv_path, 'r') as input_file:
        reader = csv.reader(input_file)
        next(reader)
        for row in reader:
            pts = list()
            # add timestamp
            timestamp = row[0]  # timestamp = row[4] + row[5].zfill(9)
            # parsing
            try:
                offset_col = row[37]
            except:
                offset_col = row[29]

            pt_cloud = np.fromstring(offset_col[1:-1], dtype=int, sep=',')
            row_without_velocity = 0
            for i in range(0, int(len(pt_cloud) / 32)):
                point = list()
                # x
                tmp = struct.pack('4B', int(pt_cloud[32 * i]), int(pt_cloud[32 * i + 1]), int(pt_cloud[32 * i + 2]),
                                  int(pt_cloud[32 * i + 3]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # y
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 4]), int(pt_cloud[32 * i + 5]), int(pt_cloud[32 * i + 6]),
                                  int(pt_cloud[32 * i + 7]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # z
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 8]), int(pt_cloud[32 * i + 9]),
                                  int(pt_cloud[32 * i + 10]),
                                  int(pt_cloud[32 * i + 11]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # intensity
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 16]), int(pt_cloud[32 * i + 17]),
                                  int(pt_cloud[32 * i + 18]),
                                  int(pt_cloud[32 * i + 19]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # range
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 20]), int(pt_cloud[32 * i + 21]),
                                  int(pt_cloud[32 * i + 22]),
                                  int(pt_cloud[32 * i + 23]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # doppler
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 24]), int(pt_cloud[32 * i + 25]),
                                  int(pt_cloud[32 * i + 26]),
                                  int(pt_cloud[32 * i + 27]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                if row_without_velocity == 1:
                    continue
                if vel_idx == vel_data.shape[0]:
                    return readings_dict
                if abs(point[0] - vel_data.loc[vel_idx]['x']) < 1e-10 and abs(
                                point[1] - vel_data.loc[vel_idx]['y']) < 1e-10 and abs(
                                point[2] - vel_data.loc[vel_idx]['z']) < 1e-10:
                    point[5] = vel_data.loc[vel_idx]['velocity']
                    vel_idx = vel_idx + 1
                else:
                    find_flag = 0
                    while vel_idx < vel_data.shape[0] - 1:
                        vel_idx = vel_idx + 1
                        if abs(point[0] - vel_data.loc[vel_idx]['x']) < 1e-10 and abs(
                                        point[1] - vel_data.loc[vel_idx]['y']) < 1e-10 and abs(
                                        point[2] - vel_data.loc[vel_idx]['z']) < 1e-10:
                            point[5] = vel_data.loc[vel_idx]['velocity']
                            vel_idx = vel_idx + 1
                            find_flag = 1
                            break
                    if find_flag == 0:
                        vel_idx = 0
                        row_without_velocity = 1
                        continue

                pts.append(point)

            readings_dict[timestamp] = pts

        return readings_dict



