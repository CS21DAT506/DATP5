import file_util as fu
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
import json 
import matplotlib.cm as cm
import file_util as FileHandler
from pathlib import Path
from progresbar import *
import scipy.stats as stats

def extract_data():
    num_of_environments = 1000

    data_dir = FileHandler.get_data_dir("testing_data_handling/testing_data")

    data = [dir for dir in FileHandler.get_data_files(data_dir) if "nn_grav" in dir ]

    extracted_data = {}

    for model in data:
        agent_time = []
        gcpd_time = []
        overhead_time = []

        for i in range(num_of_environments):
              
            folder = Path.joinpath(data_dir, model)

            archive_path = str(folder) + "/archive_" + str(i) + ".json"

            metrics = None
            with open(archive_path, "r") as file:
                json_file = file.read()
                metrics = json.loads(json_file)

            agent_time.extend(metrics["agent_time"])
            gcpd_time.extend(metrics["gcpd_time"])
            overhead_time.extend(metrics["overhead_time"])
            
            bar.next()
        print(f"\t{model} \t has been loaded")
        resetBar()
        agent_m, agent_h = mean_confidence_interval(agent_time)
        gcpd_m, gcpd_h = mean_confidence_interval(gcpd_time)
        overhead_m, overhead_h = mean_confidence_interval(overhead_time)
        extracted_data[model] = {"agent_m": agent_m, "agent_h": agent_h, 
                                 "gcpd_m": gcpd_m, "gcpd_h": gcpd_h, 
                                 "overhead_m": overhead_m, "overhead_h": overhead_h}

    with open("testing_data_handling/extracted_data.json", "w") as file:
            jsonstr = json.dumps(extracted_data, indent=4)
            file.write(jsonstr)

def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=np.float32)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)

    return m, h

if __name__ == "__main__":
    extract_data()