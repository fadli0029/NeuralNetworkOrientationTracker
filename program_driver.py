import os
import time
import yaml
from tqdm import tqdm
from modules.utils import *
from modules.preprocessing import *

def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        configs = yaml.safe_load(f)
    return configs

def update_configs(configs, args):
    if args.plot_folder:
        configs["other_configs"]["plot_figures_folder"] = args.plot_folder
    return configs

def load_datasets(datasets, configs, mode):
    print("==================================")
    print("Loading datasets...")
    print("==================================")

    data_processing_constants = configs["data_processing_constants"]
    other_configs = configs["other_configs"]

    if mode == "train":
        datasets_path = other_configs["path_to_train_datasets"]
    else:
        datasets_path = other_configs["path_to_test_datasets"]

    # load and process all imu datasets
    processed_imu_datasets = process_all_imu_datasets(
        datasets_path,
        datasets,
        data_processing_constants["vref"],
        data_processing_constants["acc_sensitivity"],
        data_processing_constants["gyro_sensitivity"],
        data_processing_constants["static_period"],
        data_processing_constants["adc_max"]
    )

    vicon_datasets = load_all_vicon_datasets(
        datasets_path,
        datasets
    )

    return processed_imu_datasets, vicon_datasets

def find_nearest(array, value):
    """
    Find the index of the nearest value in an array.

    Args:
        array: The array to search.
        value: The value to search for.

    Returns:
        The index of the nearest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def run_testing(
    tracker,
    datasets_to_test,
    processed_imu_datasets,
):
    q_optims = {}
    start = time.time()
    for dataset in datasets_to_test:
        print(f"==========> Inferencing the optimal quaternions for dataset {dataset}")

        # Unpack the processed imu data
        a_ts = processed_imu_datasets[dataset]["accs"]
        w_ts = processed_imu_datasets[dataset]["gyro"]

        data = np.hstack((a_ts, w_ts))

        # -----------------------------------------------------------------------------
        q_optims[dataset] = tracker.predict(data)
        # -----------------------------------------------------------------------------

    end = time.time()
    duration = end - start
    minutes = int(duration // 60)
    seconds = duration % 60
    print(f"==========> ✅  Done! Total duration for {len(datasets_to_test)} datasets: {minutes}m {seconds:.2f}s\n")
    return q_optims

def run_orientation_tracking(
    tracker,
    datasets_to_train,
    processed_imu_datasets,
    ground_truth_datasets,
    path_to_model
):
    start = time.time()
    datas = []
    for dataset in datasets_to_train:
        a_ts = processed_imu_datasets[dataset]["accs"]
        w_ts = processed_imu_datasets[dataset]["gyro"]
        t_ts = processed_imu_datasets[dataset]["t_ts"]

        data = np.hstack((a_ts, w_ts))
        data_ts = t_ts
        ground_truth = ground_truth_datasets[dataset]['rots']
        ground_truth_ts = ground_truth_datasets[dataset]['ts']
        indices = [find_nearest(ground_truth_ts, t) for t in data_ts]
        ground_truth = ground_truth[indices]

        datas.append((data, ground_truth))

        for _ in range(5):
            augmented_data = augment_data(data)
            datas.append((augmented_data, ground_truth))

    tracker.train(datas)

    end = time.time()
    duration = end - start
    minutes = int(duration // 60)
    seconds = duration % 60
    print(f"==========> ✅  Done! Total duration for {len(datas)} datasets: {minutes}m {seconds:.2f}s\n")

    tracker.save_model(path_to_model)

def plot_all_results(
    datasets,
    configs,
    q_optims,
    processed_imu_datasets,
    vicon_datasets,
):
    print("==================================")
    print("Saving plots...")
    print("==================================")
    other_configs = configs["other_configs"]
    pbar = tqdm(datasets, desc="==========> 📊  Saving plots", unit="plot")
    for dataset in pbar:
        iter_start = time.time()

        save_plot(
            q_optims[dataset],
            processed_imu_datasets[dataset]["accs"],
            dataset,
            vicon_datasets[dataset],
            other_configs["plot_figures_folder"],
        )
        iter_end = time.time()
        iter_duration = iter_end - iter_start

        pbar.set_postfix(time=f"{iter_duration:.4f}s")
    print(f"==========> ✅  Done! All plots saved to {other_configs['plot_figures_folder']}\n")
