# -------------------------------------------------------------------------
# Author: Muhammad Fadli Alim Arsani
# Email: fadlialim0029[at]gmail.com
# -------------------------------------------------------------------------

import argparse
from program_driver import *

from modules.neural_net import DeepLearning

def main():
    """
    """
    parser = argparse.ArgumentParser(description="Orientation tracking using IMU data.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run the algorithm.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--datasets", nargs="+", type=int, help="List of datasets to train and test the algorithm.")
    parser.add_argument("--plot_folder", type=str, help="Folder to save the plots.")

    args = parser.parse_args()

    # If datasets not provided, set the default datasets
    train_datasets = [3, 4, 5]
    test_datasets = [1, 2, 8, 9]
    if args.mode == "train":
        if args.datasets is None:
            datasets = train_datasets
        else:
            for dataset in args.datasets:
                if dataset not in train_datasets:
                    raise ValueError(f"Dataset {dataset} is not in the training datasets.")
            datasets = args.datasets
    else:
        if args.datasets is None:
            datasets = test_datasets
        else:
            for dataset in args.datasets:
                if dataset not in test_datasets:
                    raise ValueError(f"Dataset {dataset} is not in the testing datasets.")
            datasets = args.datasets

    configs = load_config(args.config)
    configs = update_configs(configs, args)

    processed_imu_datasets, vicon_datasets = load_datasets(datasets, configs, mode=args.mode)

    # Run orientation tracking
    q_optims = None
    if args.mode == "train":
        print("=====================================================")
        print(f"Training the Deep Learning model")
        print("=====================================================")
        path_to_model_folder = configs["other_configs"]["path_to_model"]

        training_parameters = configs["training_parameters"]
        tracker = DeepLearning(training_parameters)

        run_orientation_tracking(
            tracker,
            datasets,
            processed_imu_datasets,
            vicon_datasets,
            path_to_model_folder + "lstm_model.pth"
        )
    else:
        print("=====================================================")
        print(f"Testing the Deep Learning model")
        print("=====================================================")
        path_to_model_folder = configs["other_configs"]["path_to_model"]

        tracker = DeepLearning(configs["training_parameters"])
        tracker.load_model(path_to_model_folder + "lstm_model.pth")

        # run prediction on each dataset
        q_optims = run_testing(
            tracker,
            datasets,
            processed_imu_datasets,
        )

        # Save all plots
        plot_all_results(
            datasets,
            configs,
            q_optims,
            processed_imu_datasets,
            vicon_datasets,
        )

if __name__ == "__main__":
    main()
