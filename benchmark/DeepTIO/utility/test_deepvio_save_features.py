"""
Test the model using h5 files as the input and generate the the feature as the output
"""

from data_tools import load_odom_data
from networks import build_model_vanilla_vio_save_features
from keras import backend as K
from keras.models import Model
import keras
import yaml
from os.path import join, dirname
import plot_util
import math
from eulerangles import mat2euler, euler2quat, euler2mat
import time
import h5py
import numpy as np
import argparse
import inspect
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

# from utility import plot
# from utility.plot_util import plot2d

# keras

# K.set_image_dim_ordering('tf')
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))  #
# K.set_learning_phase(0)  # Run testing mode

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
K.set_image_dim_ordering('tf')
K.set_session(K.tf.Session(config=config))  #
K.set_learning_phase(0)  # Run testing mode

SCALER = 1.0  # scale label: 1, 100, 10000
RADIUS_2_DEGREE = 180.0 / math.pi


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, required=True, help='h5 file sequence, e.g. 01.h5')
    parser.add_argument('--model', type=str, required=True, help='model architecture')
    parser.add_argument('--epoch', type=str, required=True, help='which trained epoch to load in')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='specify the data dir of test data')
    parser.add_argument('--imu_length', type=str, required=True,
                        help='specify imu length')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Test directory, e.g. val, val_dark')
    parser.add_argument('--data_type', type=str, required=True, help='specify dataset type: handheld/turtle')
    parser.add_argument('--out_pred', type=str, required=False,
                        help='specify the output of csv file for the prediction)')
    parser.add_argument('--out_gt', type=str, required=False,
                        help='specify the output of csv file for the ground truth)')
    args = parser.parse_args()

    # Define and construct model
    IMU_LENGTH = int(args.imu_length)
    testest_dir = args.test_dir

    print("Building network model ......")
    deepvio_model = build_model_vanilla_vio_save_features(join('./models', args.model, args.epoch), imu_length=IMU_LENGTH,
                                                          istraining=False, base_model_name=args.model)
    deepvio_model.summary(line_length=120)

    data_dir = args.data_dir
    test_file = join(data_dir, testest_dir, args.data_type + '_seq_' + args.seq + '.h5')

    n_chunk, x_t, x_imu_t, y_t = load_odom_data(test_file, 'rgb')
    y_t = y_t[0]
    print('Data shape: ', np.shape(x_t), np.shape(y_t))
    len_x_i = x_t[0].shape[0]
    print(len_x_i)

    # Initialize value and counter
    count_img = 0
    ls_time = [0, 0, 0, 0]

    out_gt_array = []  # format (x,y) gt and (x,y) prediction
    out_pred_array = []  # format (x,y) gt and (x,y) prediction

    with K.get_session() as sess:
        print('Reading images and imu ....')
        # for i in range(0, iround ((len_thermal_x_i-2)/2)):
        for i in range(0, (len_x_i - 1)):

            # Make prediction
            st_cnn_time = time.time()
            img_1 = x_t[0][i]
            img_2 = x_t[0][i + 1]

            img_1 = np.expand_dims(img_1, axis=0)
            img_2 = np.expand_dims(img_2, axis=0)
            # img_1 = np.expand_dims(img_1, axis=0)
            # img_2 = np.expand_dims(img_2, axis=0)
            #
            imu_t = x_imu_t[0][i + 1, 0:IMU_LENGTH, :]

            imu_t = np.expand_dims(imu_t, axis=0)
            predicted = sess.run([deepvio_model.outputs],
                                 feed_dict={deepvio_model.inputs[0]: img_1,
                                            deepvio_model.inputs[1]: img_2,
                                            deepvio_model.inputs[2]: imu_t})

            pred_feat = predicted[0][0][0][0]
            print(np.shape(pred_feat))
            out_pred_array.append(pred_feat)

            count_img += 1

    if not os.path.exists('./results'):
        os.makedirs('./results')
    np.savetxt(join('./results', 'visfeatures_' + args.model + '_ep' + args.epoch + '_seq' + args.seq),
               out_pred_array, delimiter=",")

    print('Saving to h5 file ....')
    train_rgb_data_np = np.array(out_pred_array)
    train_rgb_data_np = np.expand_dims(train_rgb_data_np, axis=1)  # add dimension for time distributed

    print('Data has been collected:')
    print('RGB Features: ', np.shape(train_rgb_data_np))

    file_save = './results/rgb_feat_seq_' + args.seq + '.h5'
    with h5py.File(file_save, 'w') as hf:
        hf.create_dataset('rgb_feat', data=train_rgb_data_np)
    print('Finished! File saved in: ' + file_save)


def iround(x):
    """iround(number) -> integer
    Round a number to the nearest integer."""
    y = round(x) - .5
    return int(y) + (y > 0)


if __name__ == "__main__":
    os.system("hostname")
    main()
