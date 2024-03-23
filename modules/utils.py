# -------------------------------------------------------------------------
# Author: Muhammad Fadli Alim Arsani
# Email: fadlialim0029[at]gmail.com
# File: utils.py
# Description: This file contains utility functions for reading data and
#              saving the plot of the optimized, observed, and IMU-measured
#              acceleration data, and the RPY angles of the optimized
#              quaternion, RPY angles of the quaternions from motion model,
#              and RPY angles from VICON Motion Capture System.
# -------------------------------------------------------------------------

import os
import sys
import pickle
import numpy as np
import transforms3d as tf3d
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

def read_data(fname):
    """
    Read the data from the file and return it

    Args:
        fname (str): file name

    Returns:
        d: data
    """
    d = []
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1') # needed for python 3
    return d

def read_imu_data(dataset, path='data/'):
    """
    Read the imu data from the dataset and return it

    Args:
        dataset (int): dataset number
        path (str): path to the data folder

    Returns:
        raw_imu_data: raw imu data of shape (N, 7)
    """
    dataset = str(dataset)

    imu_file = path + "/imu/imuRaw" + dataset + ".p"

    # Reshape raw imu data to usual convention (in ML/Robotics)
    raw_imu_data = read_data(imu_file)
    raw_imu_data = np.hstack(
            (raw_imu_data['vals'].T, raw_imu_data['ts'].reshape(-1,1))
        )
    return raw_imu_data

def read_vicon_data(dataset, path='data/', quaternion=True):
    """
    Read the vicon data from the dataset and return it

    Args:
        dataset (int): dataset number
        path (str): path to the data folder

    Returns:
        vicon_data: a dictionary with keys 'rots' and 'ts',
        such that 'rots' is a numpy array of shape (N, 3, 3)
        and 'ts' is a numpy array of shape (N,)
    """
    dataset = str(dataset)
    vicon_file = path + "/vicon/viconRot" + dataset + ".p"

    # Reshape vicon data to usual convention (in ML/Robotics)
    vicon_data = read_data(vicon_file)
    vicon_data['rots'] = np.transpose(vicon_data['rots'], (2, 0, 1))

    if quaternion:
        vicon_data['rots'] = np.array([tf3d.quaternions.mat2quat(rot) for rot in vicon_data['rots']])
    vicon_data['ts'] = vicon_data['ts'].reshape(-1,)
    return vicon_data

def read_dataset(dataset, path='data/', data_name='all'):
    """
    Read the dataset and return the raw imu data and vicon data

    Args:
        dataset (int): dataset number
        path (str): path to the data folder

    Returns:
        raw_imu_data: raw imu data of shape (N, 7)
        vicon_data: a dictionary with keys 'rots' and 'ts',
                    where 'rots' is a numpy array of shape (N, 3, 3)
                    and 'ts' is a numpy array of shape (N,)
    """
    valid_data_names = ['imu', 'vicon', 'all']
    assert data_name in valid_data_names, 'Invalid data name'

    dataset = str(dataset)
    if data_name != 'all':
        if data_name == 'imu':
            return read_imu_data(dataset, path=path)
        elif data_name == 'vicon':
            return read_vicon_data(dataset, path=path, quaternion=True)
    elif data_name == 'all':
        return read_imu_data(dataset, path=path),\
               read_vicon_data(dataset, path=path)

def check_files_exist(datasets, results_folder):
    files = os.listdir(results_folder)
    files_exist = {dataset: False for dataset in datasets}
    for file in files:
        if file.endswith('.npy'):
            try:
                d = int(file.split('_')[2])
                if d in files_exist:
                    files_exist[d] = True
            except ValueError:
                continue
    return files_exist

def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    augmented_data = data + noise
    return augmented_data

def apply_random_rotation(data):
    rotation = R.random().as_matrix()
    rotated_data = np.dot(rotation, data[:, 3:].T).T  # Apply rotation to each timestep
    rotated_data = rotated_data.reshape(-1, 3)
    return np.hstack((data[:, :3], rotated_data))

def scale_data(data, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(*scale_range)
    scaled_data = data * scale_factor
    return scaled_data

def add_jitter(data, jitter_factor=0.02):
    jitter = np.random.uniform(-1, 1, data.shape) * jitter_factor
    jittered_data = data + jitter
    return jittered_data

def jumble_axes(data):
    perm = np.random.permutation(3) + 3
    while perm[0] == 3 and perm[1] == 4 and perm[2] == 5:
        perm = np.random.permutation(3) + 3
    res = (0, 1, 2, perm[0], perm[1], perm[2])
    return data[:, res]

def augment_data(data, noise_level=0.01, time_stretch_factor=0.9, jitter_factor=0.02):
    augmented_data = add_noise(data, noise_level)
    augmented_data = apply_random_rotation(augmented_data)
    augmented_data = scale_data(augmented_data)
    augmented_data = add_jitter(augmented_data, jitter_factor)
    augmented_data = jumble_axes(augmented_data)
    return augmented_data

def euler(rot):
    """
    Convert rotation matrix or quaternion to euler angles.

    Args:
        rot: rotation matrix or quaternion, shape (N, 3, 3) or (N, 4)

    Returns:
        eulers: euler angles, shape (N, 3)
    """
    if type(rot) == dict:
        rot = rot['rots']
        return np.array([tf3d.euler.quat2euler(rot[i]) for i in range(rot.shape[0])])
    return np.array([tf3d.euler.quat2euler(q) for q in rot])

def save_plot(
    q_optim,
    accs_imu,
    dataset,
    vicon_data,
    save_image_folder,
):

    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)

    # check if save_image_folder ends with '/', if not, add it
    if save_image_folder[-1] != '/':
        save_image_folder += '/'
    filename = save_image_folder + 'dataset_' + str(dataset) + '.png'

    ts = np.array(list(range(len(q_optim))))

    fig, axs = plt.subplots(3, 1, figsize=(20, 10))

    roll_ax, pitch_ax, yaw_ax = axs

    # Calculating Euler angles
    eulers_q_optim = np.array(euler(q_optim))
    eulers_vicon = np.array(euler(vicon_data))[1:]

    # Plotting the Euler angles
    roll_ax.plot(eulers_q_optim[:, 0], label='Optimized (Roll), Deep Learning', color='r')
    pitch_ax.plot(eulers_q_optim[:, 1], label='Optimized (Pitch), Deep Learning', color='r')
    yaw_ax.plot(eulers_q_optim[:, 2], label='Optimized (Yaw),  Deep Learning', color='r')

    roll_ax.plot(eulers_vicon[:, 0], label='Vicon (Roll)', color='b')
    pitch_ax.plot(eulers_vicon[:, 1], label='Vicon (Pitch)', color='b')
    yaw_ax.plot(eulers_vicon[:, 2], label='Vicon (Yaw)', color='b')

    roll_ax.set_title('Roll')
    pitch_ax.set_title('Pitch')
    yaw_ax.set_title('Yaw')
    roll_ax.legend()
    pitch_ax.legend()
    yaw_ax.legend()

    fig.savefig(filename, bbox_inches='tight')
