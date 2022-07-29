"""This module contains the data loaders for different reference systems."""

import os
import fnmatch
import pandas as pd
import numpy as np
from statistics import mean
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy.io import loadmat
from turtle import color
import matplotlib
# matplotlib.use("MacAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.LFRF_parameters.preprocessing.plot_raw_xyz import plot_acc_gyr
from src.LFRF_parameters.preprocessing.get_imu_gyro_thresholds import AccPlot, GyroPlot
from src.data_reader.DataLoader import DataLoader

from scipy.signal import find_peaks, peak_prominences

from src.LFRF_parameters.pipeline.abstract_pipeline_components import AbstractReferenceLoader
#from src.data_reader.zebris_json_reader import ZebrisJsonReader


class ZebrisReferenceLoader(AbstractReferenceLoader):
    """
    This class loads reference data created by the Zebris FDM-THQ system.
    It uses caching, since the extraction of foot positions from the Zebris raw data
    requires some computation and only needs to be done once.
    This computation is outsourced to the ZebrisJsonReader that can be also used to generate
    visualizations and inspect the data.
    """

    def load(self):
        """
        Load the data based on parameters provided in the constructor.

        Returns:
            None
        """
        # construct interim path
        self.interim_data_path = os.path.join(
            self.interim_base_path, self.dataset, self.subject, self.run
        )

        # check if cached data is present
        if (
            os.path.exists(
                os.path.join(self.interim_data_path, self.name + "_zebris_left.json")
            )
            and os.path.exists(
                os.path.join(self.interim_data_path, self.name + "_zebris_right.json")
            )
            and not self.overwrite
        ):
            self.load_interim_data()
        else:
            self.raw_data_path = os.path.join(
                self.raw_base_path, self.dataset, self.subject, self.run, "Zebris"
            )

            os.makedirs(self.interim_data_path, exist_ok=True)
            self.load_raw_data()

    def load_interim_data(self):
        """
        Load data from cached files.

        Returns:
            None
        """
        for side in self.data.keys():
            self.data[side] = pd.read_json(
                os.path.join(
                    self.interim_data_path, self.name + "_zebris_" + side + ".json"
                )
            )

    def load_raw_data(self):
        """
        Load data from raw data files.
        Zebris has two types of files (_raw and _steps).
        _raw files contain raw sensor readings.
        _steps files contain aggregated data per roll-off cycle.

        Returns:
            None
        """
        for file_name in os.listdir(self.raw_data_path):
            if fnmatch.fnmatch(file_name, "*raw.json.gz"):
                raw_json_file = os.path.join(self.raw_data_path, file_name)
            if fnmatch.fnmatch(file_name, "*steps.json.gz"):
                steps_json_file = os.path.join(self.raw_data_path, file_name)

        reader = ZebrisJsonReader(raw_json_file, steps_json_file)

        initial_contact = reader.read_zebris_raw_json_initial_contact()

        zebris_ic_fo = {}
        (
            zebris_ic_fo["left"],
            zebris_ic_fo["right"],
        ) = reader.read_zebris_raw_json_ic_fo()

        zebris_heel_positions = {}
        (
            zebris_heel_positions["left"],
            zebris_heel_positions["right"],
        ) = reader.read_zebris_raw_json_heel_positions()

        # clean up the data obtained from the zebris system
        for side in self.data.keys():
            heel_pos = zebris_heel_positions[side]

            # drop the first heel position since zebris doesn't track the time of the first rollover
            if side == "right":
                heel_pos = heel_pos[1:]

            # delete ic/fo pairs at the end
            min_steps = min(len(zebris_ic_fo[side]), len(heel_pos))
            zebris_ic_fo[side] = zebris_ic_fo[side][:min_steps]
            heel_pos = heel_pos[:min_steps]

            # make sure that the number of initial contact and foot off events matches up
            ic = zebris_ic_fo[side]["IC"].to_numpy()
            fo = zebris_ic_fo[side]["FO"].to_numpy()
            assert len(ic) == len(fo)

            # calculate actual gait parameters
            self.data[side] = pd.DataFrame(
                data={
                    "timestamp": ic[:-1] - initial_contact,
                    "stride_length_ref": heel_pos[1:] - heel_pos[:-1],
                    "stride_time_ref": ic[1:] - ic[:-1],
                    "swing_time_ref": ic[1:] - fo[:-1],
                    "stance_time_ref": fo[:-1] - ic[:-1],
                }
            )

            # save files for caching
            self.data[side].to_json(
                os.path.join(
                    self.interim_data_path, self.name + "_zebris_" + side + ".json"
                )
            )


class OptogaitReferenceLoader(AbstractReferenceLoader):
    """
    This class loads reference data created by the OptoGait system.
    """

    def load(self):
        """
        Load the data based on parameters provided in the constructor.

        Returns:
            None
        """
        self.raw_data_path = os.path.join(
            self.raw_base_path,
            # self.dataset,
            self.subject,
            self.run,
            "optogait",
            "optogait.csv",
        )

        opto_gait_data = pd.read_csv(self.raw_data_path)

        # Explanation of Optogait column names as exported by the OptoGait software:
        # # : Step index
        # L/R : left or right foots
        # TStep : step time (time between initial contacts) in s
        # Step : step length in cm
        # Split : initial contact timestamp in s
        # Stride : stride length in cm
        # StrideTime\Cycle : stride time in s
        # TStance : stance time in s
        # TSwing : swing time in s

        # select only relevant columns
        opto_gait_data = opto_gait_data[
            ["L/R", "Split", "Stride", "StrideTime\\Cycle", "TStance", "TSwing"]
        ]
        # rename columns
        opto_gait_data.rename(
            columns={
                "Split": "timestamp",
                "Stride": "stride_length_ref",
                "StrideTime\\Cycle": "stride_time_ref",
                "TStance": "stance_time_ref",
                "TSwing": "swing_time_ref",
            },
            inplace=True,
        )
        # convert cm to m
        opto_gait_data.stride_length_ref = opto_gait_data.stride_length_ref / 100

        for side in [("left", "L"), ("right", "R")]:
            # seperate right and left foot data based on "L/R" column
            self.data[side[0]] = opto_gait_data[opto_gait_data["L/R"] == side[1]].loc[
                :, opto_gait_data.columns != "L/R"
            ]

            # since optogait stores data "by contact", thus by step and not by stride, some columns are shifted.
            self.data[side[0]].stride_length_ref = self.data[
                side[0]
            ].stride_length_ref.shift(-1)
            self.data[side[0]].stride_time_ref = self.data[
                side[0]
            ].stride_time_ref.shift(-1)
            self.data[side[0]].swing_time_ref = self.data[side[0]].swing_time_ref.shift(
                -1
            )

            # drop null values
            self.data[side[0]].dropna(inplace=True)

        # zero base timestamps to the initial contact (first contact of right foot)
        self.data["left"].timestamp -= self.data["right"].timestamp.iloc[0]
        self.data["right"].timestamp -= self.data["right"].timestamp.iloc[0]


class OpticalReferenceLoader(AbstractReferenceLoader):
    def load(self):
        if "treadmill" in self.run:
            run_name = "treadmill"  # all treadmill runs use the same raw reference data
        else:
            run_name = self.run

        self.raw_data_path = os.path.join(
            self.raw_base_path,
            self.subject,
            run_name,
            "optical",
            f"omc_{run_name}.mat",
        )
        mat_data = loadmat(
            self.raw_data_path,
            simplify_cells=True)["data"]  # dictionary of data
        
        markers = [
            "l_heel",
            "l_ank",
            "l_toe",
            "l_psis",
            "r_heel",
            "r_ank",
            "r_toe",
            "r_psis"
        ]
        loc_df_list = []
        for marker in markers:
            idx = list(mat_data["marker_location"]).index(marker)  # find index for that marker
            loc_df = pd.DataFrame(  # dataframe for selected marker location
                mat_data["pos"][:,0:3,idx],  # get xyz axes for selected marker location
                columns=[
                    f"{marker}_x",
                    f"{marker}_y",
                    f"{marker}_z"
            ])
            loc_df_list.append(loc_df)
        df_loc_original = pd.concat(loc_df_list, axis=1)
        fs = mat_data["fs"]  # sampling rate 200 Hz
        df_loc_original["timestamp"] = np.arange(0, len(df_loc_original) / fs, 1 / fs)

        df_loc_original.interpolate(method="polynomial", order=2, inplace=True)  # fill missing values

        # # DEBUG: try scaling up the units to fix the stride length
        df_loc = df_loc_original.copy()
        # df_loc.iloc[:,:-1] = df_loc.iloc[:,:-1].mul(1.2)    # multiply all columns except for the timestamp

        df_loc["average_psis_x"] = df_loc[['l_psis_x', 'r_psis_x']].mean(axis=1)
        df_loc["average_psis_y"] = df_loc[['l_psis_y', 'r_psis_y']].mean(axis=1)
        df_loc["average_psis_z"] = df_loc[['l_psis_z', 'r_psis_z']].mean(axis=1)

        # for switching between left and right foot 
        self.data = {}
        for foot in [("left", "l", "r"), ("right", "r", "l")]:
            # filter raw dataframe for one foot
            df_foot = df_loc.filter(regex=f"^{foot[1]}_|average|timestamp").copy()

            # get relative distance of heel and toe along x axis (direction of walking) to psis
            df_foot[f"{foot[1]}_heel_psis_x"] = (df_foot[f"{foot[1]}_heel_x"] - df_foot["average_psis_x"]).values
            df_foot[f"{foot[1]}_heel_psis_y"] = (df_foot[f"{foot[1]}_heel_y"] - df_foot["average_psis_y"]).values
            df_foot[f"{foot[1]}_heel_psis_z"] = (df_foot[f"{foot[1]}_heel_z"] - df_foot["average_psis_z"]).values
            df_foot[f"{foot[1]}_toe_psis_x"] = (df_foot[f"{foot[1]}_toe_x"] - df_foot["average_psis_x"]).values
            df_foot[f"{foot[1]}_toe_psis_y"] = (df_foot[f"{foot[1]}_toe_y"] - df_foot["average_psis_y"]).values
            df_foot[f"{foot[1]}_toe_psis_z"] = (df_foot[f"{foot[1]}_toe_z"] - df_foot["average_psis_z"]).values

            # rotate around y axis to align the coordinated to the surface of the treadmill
            rotation_degrees = -1.3
            rotation_radians = np.radians(rotation_degrees)
            rotation_axis = np.array([0, 1, 0])

            rotation_vector = rotation_radians * rotation_axis
            rotation = R.from_rotvec(rotation_vector)
            rotated_toe = rotation.apply(df_foot[[f"{foot[1]}_toe_x", f"{foot[1]}_toe_y", f"{foot[1]}_toe_z",]].values)

            # # plot foot positions to determine direction of axes
            # fig1 = plt.figure()
            # ax = fig1.add_subplot(111, projection='3d')
            # ax.plot(df_foot[f"{foot[1]}_heel_x"], df_foot[f"{foot[1]}_heel_y"], df_foot[f"{foot[1]}_heel_z"], label="all")
            # ax.plot(df_foot[f"{foot[1]}_heel_psis_x"][1000:1200], df_foot[f"{foot[1]}_heel_psis_y"][1000:1200], df_foot[f"{foot[1]}_heel_psis_z"][1000:1200], label="1")
            # ax.plot(df_foot[f"{foot[1]}_heel_psis_x"][1200:1400], df_foot[f"{foot[1]}_heel_psis_y"][1200:1400], df_foot[f"{foot[1]}_heel_psis_z"][1200:1400], label="2")
            # ax.plot(df_foot[f"{foot[1]}_heel_psis_x"][1400:1600], df_foot[f"{foot[1]}_heel_psis_y"][1400:1600], df_foot[f"{foot[1]}_heel_psis_z"][1400:1600], label="3")
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.legend()
            # plt.show()

            # fig2 = plt.figure()
            # ax = fig2.add_subplot(111, projection='3d')
            # ax.plot(
            #     df_foot[f"{foot[1]}_toe_x"],  #[11000:12200], 
            #     df_foot[f"{foot[1]}_toe_y"],  #[11000:12200],
            #     df_foot[f"{foot[1]}_toe_z"],  #[11000:12200],
            #     label="original")
            # ax.plot(
            #     rotated_toe[:, 0],  #[11000:12200, 0], 
            #     rotated_toe[:, 1],  #[11000:12200, 1],
            #     rotated_toe[:, 2],  #[11000:12200, 2],
            #     label="rotated")
            # ax.plot(
            #     df_foot[f"{foot[1]}_toe_x"][1200:1400], 
            #     df_foot[f"{foot[1]}_toe_y"][1200:1400],
            #     df_foot[f"{foot[1]}_toe_z"][1200:1400],
            #     label="2")
            # ax.plot(
            #     df_foot[f"{foot[1]}_toe_x"][1400:1600], 
            #     df_foot[f"{foot[1]}_toe_y"][1400:1600],
            #     df_foot[f"{foot[1]}_toe_z"][1400:1600],
            #     label="3")
            # ax.plot(
            #     df_foot["average_psis_x"][1000:1200], 
            #     df_foot["average_psis_y"][1000:1200],
            #     df_foot["average_psis_z"][1000:1200],
            #     label="1")
            # ax.plot(
            #     df_foot["average_psis_x"][1200:1400], 
            #     df_foot["average_psis_y"][1200:1400],
            #     df_foot["average_psis_z"][1200:1400],
            #     label="2")
            # ax.plot(
            #     df_foot["average_psis_x"][1400:1600], 
            #     df_foot["average_psis_y"][1400:1600],
            #     df_foot["average_psis_z"][1400:1600],
            #     label="3")
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.legend()
            # plt.show()

            # # plot positions against time to check synchronization with the IMU signal
            # fig_pos = plt.figure()
            # plt.plot(df_foot[f"{foot[1]}_heel_x"], label="x")
            # plt.plot(df_foot[f"{foot[1]}_heel_y"], label="y")
            # plt.plot(df_foot[f"{foot[1]}_heel_z"], label="z")
            # plt.legend()
            # plt.show()

            # find HS and TO using peak detection
            peak_prom_threshold = 50
            # HS_list = find_peaks(dist_heel_psis, prominence = 10)[0]
            # TO_list = find_peaks(-dist_toe_psis, prominence = 10)[0]
            if run_name == "treadmill":     # the walking direction is opposite for treadmill and overground trials
                HS_list = find_peaks(-df_foot[f"{foot[1]}_heel_psis_x"], prominence = peak_prom_threshold)[0]
                TO_list = find_peaks(df_foot[f"{foot[1]}_toe_psis_x"], prominence = peak_prom_threshold)[0]
            else:
                HS_list = find_peaks(df_foot[f"{foot[1]}_heel_psis_x"], prominence = peak_prom_threshold)[0]
                TO_list = find_peaks(-df_foot[f"{foot[1]}_toe_psis_x"], prominence = peak_prom_threshold)[0]

            # find simple clearance peaks
            clearance_peak_prom_threshold = 20
            clearance_list = find_peaks(rotated_toe[:,2], prominence = clearance_peak_prom_threshold)[0]

            # # plot peaks for quality control
            # fig_peaks = plt.figure(figsize=(10, 4))
            # plt.plot(median_filter(df_foot[f"{foot[1]}_heel_psis_x"], size=20), label="heel-psis_x")
            # plt.plot(df_foot[f"{foot[1]}_heel_psis_x"], label="heel-psis_x")
            # plt.plot(df_foot[f"{foot[1]}_toe_psis_x"], label="toe-psis_x")
            # plt.plot(df_foot[f"{foot[1]}_toe_z"], label="toe_z")
            # plt.plot(rotated_toe[:,2], label="toe_z_rotated")
            # sos = signal.butter(4, 15, 'low', fs=200, output='sos')
            # z_filtered = signal.sosfilt(sos, df_foot[f"{foot[1]}_toe_z"])
            # plt.plot(z_filtered, label="toe_z_filtered")
            # plt.plot(np.gradient(np.gradient(median_filter(df_foot[f"{foot[1]}_heel_psis_x"], size=20)))*100, label="heel-psis_x acc x 100")
            # plt.plot(np.gradient(np.gradient(median_filter(df_foot[f"{foot[1]}_toe_psis_x"], size=20)))*100, label="toe-psis_x acc x 100")
            # plt.plot(HS_list, df_foot[f"{foot[1]}_heel_psis_x"][HS_list],
            #     marker='x', linestyle='None', label=f"HS, n = {len(HS_list)}")
            # plt.plot(TO_list, df_foot[f"{foot[1]}_toe_psis_x"][TO_list],
            #     marker='x', linestyle='None', label=f"TO, n = {len(TO_list)}")
            # plt.plot(clearance_list, df_foot[f"{foot[1]}_toe_z"][clearance_list],
            #     marker='x', linestyle='None', label=f"simple clearance, n = {len(clearance_list)}")
            # plt.plot(clearance_list, rotated_toe[:,2][clearance_list],
            #     marker='x', linestyle='None', label=f"simple clearance, n = {len(clearance_list)}")
            # plt.title(f"{self.subject} {foot[0]} \n gait event detection")
            # plt.ylabel("optical system z axis (mm)")
            # plt.xlabel("sample")
            # plt.legend()
            # plt.show()

            # # plot peak promineance for quality control
            # HS_prominence = peak_prominences(
            #     -df_foot[f"{foot[1]}_heel_psis_x"], HS_list
            # )[0]
            # TO_prominence = peak_prominences(
            #     df_foot[f"{foot[1]}_toe_psis_x"], TO_list
            # )[0]
            # clearance_prominence = peak_prominences(
            #     df_foot[f"{foot[1]}_toe_psis_z"], clearance_list
            # )[0]
            # fig = plt.figure()
            # plt.scatter(HS_list, HS_prominence,
            #             label='HS, n=' + str(len(HS_list)) + ', threshold = ' + str(peak_prom_threshold),
            #             c='orange')
            # plt.scatter(TO_list, TO_prominence,
            #             label='TO, n=' + str(len(TO_list)) + ', threshold = ' + str(peak_prom_threshold),
            #             c='darkturquoise')
            # # plt.scatter(clearance_list, clearance_prominence,
            # #             label='simple clearance, n=' + str(len(clearance_list)) + ', threshold = ' + str(clearance_peak_prom_threshold),
            # #             c='darkturquoise')
            # plt.axhline(y=clearance_peak_prom_threshold, color='coral', linestyle='-')
            # plt.title(f'{self.subject} {foot[0]} \n prominences of HS and TO peaks')
            # plt.xlabel('sample number')
            # plt.ylabel('prominence')
            # plt.legend()
            # plt.show()

            # create boolean array to document HS and TO events
            HS_boolean = np.full(len(df_foot), fill_value = False)
            TO_boolean = np.full(len(df_foot), fill_value = False)

            for i in HS_list:    #iterate over array and insert TRUE for peaks
                HS_boolean[i] = True

            for i in TO_list:
                TO_boolean[i] = True

            #save Boolean column in DF
            df_foot["HS"] = HS_boolean
            df_foot["TO"] = TO_boolean
            
            filter_df = df_foot[(df_foot["HS"]== True) | (df_foot["TO"] == True)]   # get only the gait events
            while not filter_df["HS"].iloc[0]:       # the event sequence should always start with HS
                filter_df.drop(filter_df.index[0], inplace = True)
            while not filter_df["TO"].iloc[1]:       # and followed by a TO event
                filter_df.drop(filter_df.index[1], inplace = True)

            diff_filter_df = filter_df["TO"].astype(int).diff()

            # correct for missing HS or TO events
            # remove first false if we ahve doule false
            # remove second true for double true

            # keep the last 0 if previous is -1 and remove -1 -> set in the for loop the current 0 to -1 
            # remove all 0s if previous is 1

            diff_filter_list = diff_filter_df.index.astype("Int64").tolist()

            for i in range(1, len(diff_filter_list)):
                if diff_filter_df[diff_filter_list[i]] == 0 and diff_filter_df[diff_filter_list[i-1]] == 1:
                    diff_filter_df[diff_filter_list[i]] = np.nan

            
                elif diff_filter_df[diff_filter_list[i]] == 0 and diff_filter_df[diff_filter_list[i-1]] == -1:
                    diff_filter_df[diff_filter_list[i]] = -1
                    diff_filter_df[diff_filter_list[i-1]] = np.nan 

                elif diff_filter_df[diff_filter_list[i]] == 0 and np.isnan(diff_filter_df[diff_filter_list[i-1]]):
                    diff_filter_df[diff_filter_list[i]] = np.nan


            diff_filter_df.dropna(inplace = True)
            df_final = df_foot.loc[diff_filter_df.index]

            if df_final["TO"].iloc[0]:
                df_final.drop(df_final.index[0], inplace = True)

            if df_final["TO"].iloc[-1]:
                df_final.drop(df_final.index[-1], inplace = True)

            df_final.reset_index(inplace=True, drop= True)

            ###################### Calculating Gait parameters  ########################

            #### stance time -> HS to TO, swing time -> TO to HS ##################
            time_HS = df_final.query('HS == True', inplace = False)['timestamp']
            time_TO = df_final.query('TO == True', inplace = False)['timestamp']

            time_HS.reset_index(inplace=True, drop=True)
            time_TO.reset_index(inplace=True, drop=True)

            stance_time = np.array(time_TO) - np.array(time_HS[:-1])
            swing_time = np.array(time_HS[1:]) - np.array(time_TO)

            #### calculate stride length using step length #################

            # Merge the DataFrames to include positions from the other foot
            LR_merged = pd.merge(df_loc, df_final, how='inner', 
                                    on="timestamp", suffixes=('', '_remove'))
            # remove the duplicate columns
            LR_merged.drop([i for i in LR_merged.columns if 'remove' in i], 
                            axis=1, inplace=True)

            value_HS = LR_merged.query('HS == True', inplace = False).copy()
            value_TO = LR_merged.query('TO == True', inplace = False).copy()

            # calculate step length: distance from heel (foot L/R) to the other heel (foot R/L)
            step_length1 = [
                (value_HS[f"{foot[1]}_heel_x"] - value_HS[f"{foot[2]}_heel_x"])[1:].values,
                (value_HS[f"{foot[1]}_heel_y"] - value_HS[f"{foot[2]}_heel_y"])[1:].values,
                (value_HS[f"{foot[1]}_heel_z"] - value_HS[f"{foot[2]}_heel_z"])[1:].values
                ]   # step length current foot in front
            step_length2 = [
                (value_TO[f"{foot[2]}_heel_x"] - value_TO[f"{foot[1]}_heel_x"]).values,
                (value_TO[f"{foot[2]}_heel_y"] - value_TO[f"{foot[1]}_heel_y"]).values,
                (value_TO[f"{foot[2]}_heel_z"] - value_TO[f"{foot[1]}_heel_z"]).values
                ]    # step length current foot at back
            stride_length = np.linalg.norm(np.add(
                np.asarray(step_length1), 
                np.asarray(step_length2)
                ), axis=0)  # norm of x, y and z axis displacement
            stride_length = stride_length / 1000  # convert mm to meter

            #### calculate stride time ################
            stride_time = time_HS.diff()
            stride_time.dropna(inplace=True)    # drop index 0 which is nan

            #### calculate simple clearance ####
            # shift to ground level
            # relative_z = rotated_toe[:,2] - mean(rotated_toe[10:600, 2])
            relative_z = rotated_toe[:,2] - mean(df_foot[f"{foot[1]}_toe_z"][10:600])

            clearance = []
            for swing_start, swing_end in zip(
                    time_TO.values,
                    time_HS[1:].values,
            ):
                clearance_cand = list(x for x in clearance_list if int(swing_start*fs) <= x <= int(swing_end*fs))
                if not (np.isnan(clearance_cand).any() or len(clearance_cand) == 0):
                    clearance_values = relative_z[clearance_cand]  # .values
                    clearance_idx = clearance_values.argmax()  # identify the largest peak if there are multiple ones
                    peak_value = clearance_values[clearance_idx] / 1000  # get clearance relative to ground level, convert mm to m
                    clearance.append(peak_value)
                else:
                    clearance.append(np.nan)  # placeholder to match other gait parameters

            #### summarize all gait parameters ####
            data = {
                "timestamp":time_HS[:-1].values,
                "ic_time_ref":time_HS[1:].values,
                "fo_time_ref":time_TO.values, 
                "stance_time_ref": stance_time, 
                "swing_time_ref": swing_time, 
                "stride_length_ref": stride_length,
                "stride_time_ref": stride_time.values,
                "clearance_ref": clearance
                } 
            self.data[foot[0]] = pd.DataFrame.from_dict(data)

            # No need to modify the timestamps when using the entire recording,
            # if the IMU and optical systems are already synched.
            # Otherwise, if using only part of the recording:
            # zero base timestamps according to metadata

            if "speed" in self.run:
                #for treadmill: load timestamps of start for constant speed to match with the IMU data
                speed_timestamps_df = pd.read_csv(os.path.join(self.raw_base_path, "treadmill_timestamps.csv"))
                start = speed_timestamps_df[speed_timestamps_df["sub"] == self.subject].filter([f"start{self.run[-1]}"]).values[0]  # timestamp in seconds
                self.data[foot[0]].timestamp -= start

