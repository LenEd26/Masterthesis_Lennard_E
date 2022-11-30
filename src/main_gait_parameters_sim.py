import json
import os
from src.LFRF_parameters import pipeline_playground 


### PARAMS START ###

dataset = "data_charite"


if dataset == "data_charite":
    stroke_list = [
        #"imu0001",
        #"imu0002",
        #"imu0003",
        "imu0006"
    ]

    runs = [
        #"visit1",
        "visit2"
    ]


if dataset == "data_sim":
    stroke_list = [
        "S1",
        "S2"
    ]


    runs = [
        "1P",
        "2P",
        "3P",
        "regular"

    ]
elif dataset == "data_sim_cbf":
    stroke_list = ["S1"]
    
    runs = [
        #"leicht", 
        "leicht2",
        #"leicht3",
        #"normal",
        #"stark"   
    ]

elif dataset == "data_stanford":
    stroke_list = ["S1"  
                 ]
    
    runs = [
        "NLA"  ]

with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
data_base_path = paths[dataset]
## PARAMS END ###

print("basepath_____",data_base_path)
### Execute the Gait Analysis Pipeline ###
pipeline_playground.execute(stroke_list, runs, dataset, data_base_path)
