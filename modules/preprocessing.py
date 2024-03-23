# -------------------------------------------------------------------------
# Author: Muhammad Fadli Alim Arsani
# Email: fadlialim0029[at]gmail.com
# File: preprocessing.py
# Description: This file contains functions to process the raw IMU data
#              to get acceleration and angular velocity in physical units.
# Misc: This is also part of one of the projects in the course
#       "Sensing and Estimation in Robotics" taught by Prof. Nikolay
#       Atanasov @UC San Diego.
#       https://natanaso.github.io/ece276a/index.html
# -------------------------------------------------------------------------

import time
import numpy as np
from .utils import *
from tqdm import tqdm

def process_imu_data(
    imu_data_raw,
    vref,
    acc_sens,
    gyro_sens,
    static_period,
    adc_max
):
    """
    Process raw IMU data to get acceleration
    and angular velocity in physical units.

    Args:
        imu_data_raw:  raw IMU data, shape (N, 7), last column is
                       timestamp, otherwise:
                       imu_data_raw[:, :3]  -> acc
                       imu_data_raw[:, 3:6] -> gyro

        vref:          reference voltage of the IMU
        acc_sens:      sensitivity of the accelerometer
        gyro_sens:     sensitivity of the gyroscope
        static_period: period of time to compute the bias
        adc_max:       maximum value of the ADC

    Returns:
        acc_data: acceleration in physical units
        gyro_data: angular velocity in physical units
        timestamp: timestamp of the data
    """
    # convert constants to appropiate units
    vref = vref*1000
    gyro_sens = gyro_sens*(180./np.pi)

    # extract the static portion of the data
    static_data = imu_data_raw[
        imu_data_raw[:, -1] <= (imu_data_raw[0, -1] + static_period)
    ]

    # compute the mean of the static portion
    bias = np.mean(static_data[:, :-1], axis=0) # shape (6,)

    # Compute scale factors
    acc_sf = (vref/adc_max)/acc_sens
    gyro_sf = (vref/adc_max)/gyro_sens

    # Bias correction and conversion for acc. data
    acc_data = (imu_data_raw[:, :3] - bias[:3])*acc_sf

    # Axis correction (due to device design)
    acc_data[:, 0] = -acc_data[:, 0]
    acc_data[:, 1] = -acc_data[:, 1]

    # add gravity since we expect Az to be g during static period
    acc_data[:, 2] += 1.0

    # Bias correction and conversion for gyro. data
    gyro_data = (imu_data_raw[:, 3:6] - bias[3:])
    gyro_data[:, [0, 1, 2]] = gyro_data[:, [1, 2, 0]]*gyro_sf

    return acc_data, gyro_data, imu_data_raw[:, -1]

def process_all_imu_datasets(
    path: str,
    datasets: list,
    vref,
    acc_sens,
    gyro_sens,
    static_period,
    adc_max
):
    """
    Process all IMU datasets.

    Args:
        path:          path to the datasets
        datasets:      list of datasets to process
        vref:          reference voltage of the IMU
        acc_sens:      sensitivity of the accelerometer
        gyro_sens:     sensitivity of the gyroscope
        static_period: period of time to compute the bias
        adc_max:       maximum value of the ADC

    Returns:
        processed_imu_data: dictionary with the processed data
                            where the key is the dataset name and
                            the value is a dictionary with the
                            acceleration, angular velocity and
                            timestamp.
    """
    assert len(datasets) > 0, "No datasets provided!"
    assert type(datasets) == list, "Datasets must be a list!"

    print(f"==========> Processing {len(datasets)} IMU datasets.")
    start = time.time()
    processed_imu_data = {}
    for dataset in tqdm(datasets, desc="Processing IMU datasets"):
        # load the dataset
        imu_data_raw = read_dataset(
            dataset, path=path, data_name='imu'
        )

        # process the raw data
        acc_data, gyro_data, timestamps = process_imu_data(
            imu_data_raw,
            vref,
            acc_sens,
            gyro_sens,
            static_period,
            adc_max
        )
        processed_imu_data[dataset] = {
            "accs": acc_data,
            "gyro": gyro_data,
            "t_ts": timestamps
        }
    duration = time.time() - start
    print(f"Done! Took {duration:.2f} seconds.\n")
    return processed_imu_data

def load_all_vicon_datasets(path: str, datasets: list):
    """
    Load all Vicon datasets.

    Args:
        path:     path to the datasets
        datasets: list of datasets to load

    Returns:
        vicon_datasets: dictionary with the loaded data
        where the key "rots" of shape (N, 3, 3) is the
        rotation matrix and the key "ts" of shape (N,)
        is the timestamp when the data was recorded.
    """
    assert len(datasets) > 0, "No datasets provided!"
    assert type(datasets) == list, "Datasets must be a list!"

    print(f"==========> Loading {len(datasets)} Vicon datasets.")
    start = time.time()
    vicon_datasets = {}
    for dataset in tqdm(datasets, desc="Loading VICON datasets"):
        vicon_datasets[dataset] = read_dataset(dataset, path=path, data_name='vicon')
    duration = time.time() - start
    print(f"Done! Took {duration:.2f} seconds.\n")
    return vicon_datasets
