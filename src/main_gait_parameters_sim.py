import json
import os
from src.LFRF_parameters import pipeline_playground 


### PARAMS START ###


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
dataset = 'data_sim'
with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
data_base_path = paths[dataset]
## PARAMS END ###

### Execute the Gait Analysis Pipeline ###
pipeline_playground.execute(stroke_list, runs, dataset, data_base_path)

## aggregation of datasets