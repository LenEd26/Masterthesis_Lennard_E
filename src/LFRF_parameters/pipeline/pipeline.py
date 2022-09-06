"""This module contains the pipeline class."""

import os
import numpy as np
import pandas as pd
from scipy.stats import variation

# all non-variable parts of the pipeline need to be imported
from src.data_reader.imu import IMU
from src.LFRF_parameters.pipeline.gait_parameters import GaitParameters
from src.LFRF_parameters.pipeline.evaluator import Evaluator
# from features.aggregate_gait_parameters import aggregate_parameters
from src.visualisation.plot import plot_multi_3d_view, show, plot_accel_gyro


class Pipeline:
    """
    This class is the skeleton for all possible pipeline instantiations.

    It is instantiated with a config dictionary that defines the different variable components
    of the pipeline, the data source and auxillary variables.
    """

    def __init__(self, pipeline_config):
        """Instantiation of a pipeline.
        For the necessary variables and explanation of the configuration dictionary see pipeline_playground.py.

        Args:
            pipeline_config (dict): Dictionary containing configuration variables
        """
        self.config = pipeline_config
        self.imu_ic = None
        self.imu_gyro_threshold = None
        #self.evaluator = Evaluator()
        
        #load initial contacts from csv file
        self.imu_ic_timestamps = pd.read_csv(
            os.path.join(
                self.config["interim_base_path"],
                "imu_initial_contact_manual.csv",
            )
        )

        #load gyro thresholds from csv file
        self.imu_gyro_thresholds = pd.read_csv(
            os.path.join(
                self.config["interim_base_path"],
                "stance_magnitude_thresholds_manual.csv",
            )
        )

    def load_data(self, subject_num, run_num):
        """
        Load all IMU data for the given subject_num and run_num from the IMU folder.

        Args:
            subject_num (int): Index of the subject whose data should be loaded

            run_num (int): Index of the run/trial that should be loaded

        Returns:
            tuple[dict[str, IMU], float): Tuple of a dictionary of IMU objects and the timestamp of initial contact in seconds
        """

        folder_path = os.path.join(
            self.config['interim_base_path'],
            self.config['subjects'][subject_num],
            self.config['runs'][run_num],
            "imu"
        )
        imus = {}
        for kw in self.config["location_kws"]:
            IMU_path = os.path.join(folder_path, kw + '.csv')
            imu = IMU(IMU_path)  # read interim IMU data
            imu.acc_to_meter_per_square_sec()
            imu.gyro_to_rad()
            # plot_accel_gyro(imu)
            # show()
            imus[kw] = imu

        # imus = self.config["data_loader"](
        #     self.config["raw_base_path"],
        #     self.config["dataset"],
        #     self.config["subjects"][subject_num],
        #     self.config["runs"][run_num],
        # ).get_data()

        imu_ic = float(
            self.imu_ic_timestamps[
                np.logical_and(
                    self.imu_ic_timestamps["subject"]
                    == self.config["subjects"][subject_num],
                    self.imu_ic_timestamps["run"] == self.config["runs"][run_num],
                )
            ]["imu_initial_contact_right"]   # ic_time
        )

        # crop imu data to fit experiment_duration seconds from inititial contact
        # start 2 seconds earlyer to get also lift-off data before initial contact
        # for imu in imus.values():
        #     imu.crop(
        #         imu_ic - 2,
        #         imu_ic + self.config["experiment_duration"],
        #         inplace=True,
        #     )

        self.stance_thresholds = self.imu_gyro_thresholds[
            np.logical_and(
                self.imu_gyro_thresholds["subject"]
                == self.config["subjects"][subject_num],
                self.imu_gyro_thresholds["run"] == self.config["runs"][run_num],
            )
        ][
            [
                "stance_magnitude_threshold_left",
                "stance_magnitude_threshold_right",
                "stance_count_threshold_left",
                "stance_count_threshold_right",
            ]
        ]

        return imus, imu_ic

    def detect_gait_events(self, subject_num, run_num, imus, trajectories, save_fig_directory):
        """Perform gait event detection with the gait event detector specified in the config.

        Args:
            imus (dict[str, IMU]): IMU objects for each sensor location

        Returns:
            dict[str, dict]: IC and FO samples and timestamps for the right and left foot
        """
        return self.config["gait_event_detector"](imus).detect(
            self.stance_thresholds,
            self.config["interim_base_path"],
            self.config["dataset"],
            self.config["subjects"][subject_num],
            self.config["runs"][run_num],
            self.config["prominence_search_threshold"],
            self.config["prominence_ic"],
            self.config["prominence_fo"],
            self.config["show_figures"],
            trajectories,
            save_fig_directory
        )

    def estimate_trajectories(self, subject_num, run_num, imus, save_fig_directory, imu_ic=0):
        """
        Estimate left and right foot trajectories using the specified trajectory estimation algorithm.
        Subject and run need to be identified since the trajectory estimator uses caching.

        Args:
            subject_num (int): subject index
            run_num (int): run index
            imus (dict[str, IMU]):  IMU objects for each sensor location
            imu_ic (float): Inital contact timestamp

        Returns:
            dict[str, DataFrame]: DataFrames with trajectory information for the right and left foot
        """
        return self.config["trajectory_estimator"](imus).estimate(
            # self.config["name"],
            self.config["interim_base_path"],
            self.config["dataset"],
            self.config["subjects"][subject_num],
            self.config["runs"][run_num],
            imu_ic,
            self.stance_thresholds,
            self.config["sampling_rate"],
            self.config["overwrite"],
            self.config["show_figures"],
            save_fig_directory
        )

    def calculate_gait_parameters(self, subject_num, run_num, gait_events, trajectories, save_fig_directory, 
    save = True, imu_ic=0):
        """Calculate gait parameters.

        Args:
            gait_events (dict[str, dict]): IC and FO samples and timestamps for the right and left foot
            trajectories (dict[str, DataFrame]): DataFrames with trajectory information for the right and left foot
            imu_ic (float): Inital contact timestamp

        Returns:
            dict[str, DataFrame]: DataFrame with the estimated gait parameters for the left and right foot
        """

        summary = GaitParameters(trajectories,
                                 gait_events,
                                 self.config['show_figures'],
                                 save_fig_directory,
                                 imu_ic).summary()

        # if save:
        save_path = os.path.join(
                self.config["processed_base_path"],
                self.config["subjects"][subject_num],
                self.config["runs"][run_num]
            )
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)

        summary["left"].to_csv(os.path.join(save_path, "left_foot_core_params_py_n.csv"), index=False)
        summary["right"].to_csv(os.path.join(save_path, "right_foot_core_params_py_n.csv"), index=False)
            #aggregate["left"].to_csv(os.path.join(save_path, "left_foot_aggregate_params_py_n.csv"), index=False)
            #aggregate["right"].to_csv(os.path.join(save_path, "right_foot_aggregate_params_py_n.csv"), index=False)
        print("saved gait parameters for " +
            self.config["runs"][run_num] + ", " +
            self.config["subjects"][subject_num])

        return summary

    # def load_reference_data(self, subject_num, run_num):
    #     """Load reference data with the specified reference loader.
    #     Note: subject and run need to be identified since the reference loader uses caching.

    #     Args:
    #         subject_num (int): subject index
    #         run_num (int): run index

    #     Returns:
    #         dict[str, DataFrame]: DataFrames with gait parameters for the left and right foot
    #     """

    #     return self.config["reference_loader"](
    #         self.config["name"],
    #         self.config["raw_base_path"],
    #         self.config["interim_base_path"],
    #         self.config["dataset"],
    #         self.config["subjects"][subject_num],
    #         self.config["runs"][run_num],
    #         self.config["overwrite"],
    #     ).get_data()

    # def add_to_evaluator(self, subject_num, run_num, gait_parameters, reference_data):  #
    #     """Add estimated gait parameters and reference data for one subject and run to the evaluator.

    #     Args:
    #         subject_num (int): subject index
    #         run_num (int): run index
    #         reference_data (dict[str, DataFrame]): DataFrames with reference gait parameters for the left and right foot
    #         gait_parameters (dict[str, DataFrame]): DataFrame with the estimated gait parameters for the left and right foot

    #     Returns:
    #         None
    #     """

    #     self.evaluator.add_data(subject_num, run_num, gait_parameters) #, reference_data)

    def execute(self, subject_runs):
        """
        Core function of the pipeline.
        For each subject and run:
        Load IMU data, estimate trajectories, detect gait events,
        calculate estimated gait parameters, add them together with the
        reference_data to the evaluator.
        Exectue the evaluator with the results of all runs altogether.

        Args:
            subject_runs (tuple[int, int]): Index of the subject and run whose data should be loaded

        Returns:
            None
        """
        # executes all pipeline stages
        for subject_num, run_num in subject_runs:
            base_file_directory = os.path.join(self.config["processed_base_path"],
                                              # self.config["dataset"],
                                              )
            if not os.path.exists(base_file_directory):
                os.makedirs(base_file_directory)

            save_fig_dir = os.path.join(base_file_directory,
                                              'pipeline_figures',
                                              self.config["subjects"][subject_num],
                                              self.config["runs"][run_num]
                                              )
            if not os.path.exists(save_fig_dir):
                os.makedirs(save_fig_dir)

            aggregate_params_dir = os.path.join(base_file_directory,
                                                self.config["subjects"][subject_num],
                                                self.config["runs"][run_num]
                                                )
            if not os.path.exists(aggregate_params_dir):
                os.makedirs(aggregate_params_dir)

            print(
                "processing run",
                self.config["runs"][run_num],
                "subject",
                self.config["subjects"][subject_num],
            )

            print("load data")
            imu_data, imu_ic = self.load_data(subject_num, run_num)

            print("estimate trajectories")
            trajectories = self.estimate_trajectories(
                subject_num, run_num, imu_data, save_fig_dir
            )

            print("detect gait events")
            gait_events = self.detect_gait_events(
                subject_num, run_num, imu_data, trajectories, save_fig_dir
            )

            print("calculate gait parameters")
            gait_parameters = self.calculate_gait_parameters(
                subject_num, run_num, gait_events, trajectories, save_fig_dir, imu_ic
            )
            print(gait_parameters)


            #print('calculate aggregate_parameters')
            #aggregate_params, _ = aggregate_parameters(aggregate_params_dir, save=True)

            # print("load reference data")
            # reference_data = self.load_reference_data(subject_num, run_num)

            #self.add_to_evaluator(subject_num, run_num, gait_parameters) #, reference_data
            #print()

        # match reference system and estimated gait parameters stride by stride
        # self.evaluator.match_timestamps()

        # generate plots
        # self.evaluator.plot_correlation(
        #     "Tunca et al.", "stride_length", subject_runs, self.config["reference_name"]
        # )
        # self.evaluator.plot_correlation(
        #     "Tunca et al.", "clearance", subject_runs, self.config["reference_name"]
        # )
        # self.evaluator.plot_correlation(
        #     "Tunca et al.", "stride_time", subject_runs, self.config["reference_name"]
        # )
        # self.evaluator.plot_bland_altmann(
        #     "stride_length", subject_runs, self.config["reference_name"]
        # )
        # self.evaluator.plot_bland_altmann(
        #     "clearance", subject_runs, self.config["reference_name"]
        # )
        # self.evaluator.plot_bland_altmann(
        #     "stride_time", subject_runs, self.config["reference_name"]
        # )

        # self.evaluator.plot_correlation(
        #     "Tunca et al.", "swing_time", subject_runs, self.config["reference_name"]
        # )
        # self.evaluator.plot_correlation(
        #     "Tunca et al.", "stance_time", subject_runs, self.config["reference_name"]
        # )
        # self.evaluator.plot_bland_altmann(
        #     "swing_time", subject_runs, self.config["reference_name"]
        # )
        # self.evaluator.plot_bland_altmann(
        #     "stance_time", subject_runs, self.config["reference_name"]
        # )