import json
import os
import sys

from src.LFRF_parameters.pipeline.data_loader import *
from src.LFRF_parameters.pipeline.event_detector import TuncaEventDetector, LaidigEventDetector
from src.LFRF_parameters.pipeline.trajectory_estimator import TuncaTrajectoryEstimator
from src.LFRF_parameters.pipeline.reference_loader import OptogaitReferenceLoader
from src.LFRF_parameters.pipeline.reference_loader import OpticalReferenceLoader
from src.LFRF_parameters.pipeline.pipeline import Pipeline

def execute(sub_list, runs, dataset, data_base_path):
    """
    Executes the Playground pipeline.
    Returns
    -------

    """
    #configure the pipeline
    if dataset == "data_kiel" or dataset == "data_kiel_val":
        pipeline_config = {
                            # @name: the name should be unique for each pipeline configuration.
                            # it is used to identify interim data and reuse it in the next run
                            "name": "data_kiel",
                            'raw_base_path': os.path.join(data_base_path, "raw"),
                            'interim_base_path': os.path.join(data_base_path, "interim"),
                            'processed_base_path': os.path.join(data_base_path, "processed"),
                            'overwrite': False,  # overwrite the trajectory estimations
                            'show_figures': 0,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
                            'location_kws': ['LF', 'RF'],
                            'data_loader': PhysilogDataLoader,
                            'trajectory_estimator': TuncaTrajectoryEstimator,
                            'sampling_rate': 200,
                            'gait_event_detector': TuncaEventDetector, # LaidigEventDetector,
                            'prominence_search_threshold': 0.3,
                            'prominence_ic': 0.1,
                            'prominence_fo': 0.3,
                            "reference_loader": OpticalReferenceLoader,
                            "reference_name": "OpticalSystem",
                            'dataset': dataset,
                            'runs': runs,
                            'subjects': sub_list,
        }

    elif dataset == "data_sim" or dataset == "data_sim_cbf":
        pipeline_config = {
                            # @name: the name should be unique for each pipeline configuration.
                            # it is used to identify interim data and reuse it in the next run
                            "name": "data_sim",
                            'raw_base_path': os.path.join(data_base_path, "raw"),
                            'interim_base_path': os.path.join(data_base_path, "interim"),
                            'processed_base_path': os.path.join(data_base_path, "processed"),
                            'overwrite': False,  # overwrite the trajectory estimations
                            'show_figures': 0,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
                            'location_kws': ['LF', 'RF'],
                            'data_loader': PhysilogDataLoader,
                            'trajectory_estimator': TuncaTrajectoryEstimator,
                            'sampling_rate': 120, 
                            'gait_event_detector': TuncaEventDetector,
                            'prominence_search_threshold': 0.7,
                            'prominence_ic': 0.01,
                            'prominence_fo': 0.01,
                            "reference_loader": OptogaitReferenceLoader,
                            "reference_name": "OptoGait", 
                            'dataset': dataset,
                            'runs': runs,
                            'subjects': sub_list,
        }

    elif dataset == "data_stanford":
        pipeline_config = {
                        # @name: the name should be unique for each pipeline configuration.
                        # it is used to identify interim data and reuse it in the next run
                        "name": "data_stanford",
                        'raw_base_path': os.path.join(data_base_path, "raw"),
                        'interim_base_path': os.path.join(data_base_path, "interim"),
                        'processed_base_path': os.path.join(data_base_path, "processed"),
                        'overwrite': False,  # overwrite the trajectory estimations
                        'show_figures': 0,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
                        'location_kws': ['LF', 'RF'],
                        'data_loader': PhysilogDataLoader,
                        'trajectory_estimator': TuncaTrajectoryEstimator,
                        'sampling_rate': 100,
                        'gait_event_detector': TuncaEventDetector,
                        'prominence_search_threshold': 0.3,
                        'prominence_ic': 0.01,
                        'prominence_fo': 0.01,
                        "reference_loader": OptogaitReferenceLoader,
                        "reference_name": "OptoGait", ##?
                        'dataset': dataset,
                        'runs': runs,
                        'subjects': sub_list,
    }

    elif dataset == "data_imu_validation":
        pipeline_config = {
                            # @name: the name should be unique for each pipeline configuration.
                            # it is used to identify interim data and reuse it in the next run
                            "name": "data_imu_validation",
                            'raw_base_path': os.path.join(data_base_path, "raw"),
                            'interim_base_path': os.path.join(data_base_path, "interim"),
                            'processed_base_path': os.path.join(data_base_path, "processed"),
                            'overwrite': False,  # overwrite the trajectory estimations
                            'show_figures': 1,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
                            'location_kws': ['LF', 'RF'],
                            'data_loader': PhysilogDataLoader,
                            'trajectory_estimator': TuncaTrajectoryEstimator,
                            'sampling_rate': 120,
                            'gait_event_detector': TuncaEventDetector,
                            'prominence_search_threshold': 0.7,
                            'prominence_ic': 0.01,
                            'prominence_fo': 0.01,
                            "reference_loader": OptogaitReferenceLoader,
                            "reference_name": "OptoGait",
                            'dataset': dataset,
                            'runs': runs,
                            'subjects': sub_list,
        }

    elif dataset == "data_TRIPOD":
        pipeline_config = {
                            # @name: the name should be unique for each pipeline configuration.
                            # it is used to identify interim data and reuse it in the next run
                            "name": "data_TRIPOD",
                            'raw_base_path': os.path.join(data_base_path, "raw"),
                            'interim_base_path': os.path.join(data_base_path, "interim"),
                            'processed_base_path': os.path.join(data_base_path, "processed"),
                            'overwrite': False,  # overwrite the trajectory estimations
                            'show_figures': 0,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
                            'location_kws': ['LF', 'RF'],
                            'data_loader': PhysilogDataLoader,
                            'trajectory_estimator': TuncaTrajectoryEstimator,
                            'sampling_rate': 128,
                            'gait_event_detector': TuncaEventDetector,
                            'prominence_search_threshold': 0.7,
                            'prominence_ic': 0.01,
                            'prominence_fo': 0.03,
                            "reference_loader": OptogaitReferenceLoader,
                            "reference_name": "OptoGait",
                            'dataset': dataset,
                            'runs': runs,
                            'subjects': sub_list,
        }

    # create the pipeline
    pipeline = Pipeline(pipeline_config)
    print("pipeine______:", pipeline)

    # list of tuples (run number, subject number)
    everything = [(x, y) for x in range(0, len(pipeline_config["subjects"])) for y in range(0, len(pipeline_config["runs"]))]
    # analyze = [(1, 0), (1, 1), (1, 2)]

    print("everything___:",everything)

    analyze = everything
    pipeline.execute(analyze)

