import json
import os
from src.LFRF_parameters import pipeline_playground 


### PARAMS START ###
control_list = [
    "pp010",
    "pp011",
    "pp028",
    # "pp071",    # part of the RF data is not usable (IMU dropped from the foot)
    "pp079",
    "pp099",
    "pp105", 
    "pp106",
    # "pp114",    # only one constant treadmill speed
    "pp137",
    "pp139", 
    "pp158",
    "pp165",
    # "pp166"     # only one constant treadmill speed
]

stroke_list = [
    #"pp077",
    "pp105", #?? why no timestamps etc.
    # "pp107",
    #"pp111",
    #"pp122",
    # "pp127",
    # "pp136",
    #"pp152"
]


runs = [
    "treadmill"      # all treadmill data, including changing speed
    # "treadmill_speed1",     # constant speed 1
    # "treadmill_speed2",     # constant speed 2
    # "gait1",
    # "gait2",
    # "walk_fast",
    #"walk_preferred",
    # "walk_slow"
]
dataset = 'data_kiel'
with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
data_base_path = paths[dataset]
## PARAMS END ###

### Execute the Gait Analysis Pipeline ###
pipeline_playground.execute(stroke_list, runs, dataset, data_base_path)


