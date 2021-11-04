from nn_util import load_nn_data
from tf_main import get_data_dir, get_data_files
from pathlib import Path
import json
import numpy as np
from progress.bar import IncrementalBar

def get_agent_gravity(agent_pos, planets, G=6.646596924499661e-05):
    agent_acc = np.array((0, 0))
    agent_pos = np.array(agent_pos)

    for planet in planets:
        distance = np.array(planet[:2]) - agent_pos
        agent_acc = agent_acc + planet[-1] * distance / np.linalg.norm(distance)**3

    return agent_acc * G

if __name__ == '__main__':
    data_dir = get_data_dir("nn_trainer/data")
    data_files = get_data_files(data_dir)

    new_data_dict = {
        "input": [],
        "output": [],
    }

    bar = IncrementalBar("processing",max=len(data_files)+1, suffix='%(percent)d%%')

    for file in data_files:
        bar.next()

        path_to_json_file = str(Path.joinpath(data_dir, file))
        data_input, data_output = load_nn_data(path_to_json_file)

        new_data_dict["output"].extend([data_point[0] for data_point in data_output])

        for data in data_input:
            data = data[0]
            new_data_input = data[:6]

            agent_pos = new_data_input[2:4]
            planets = [data[i:i+5] for i in range(7,len(data),5)]

            new_data_input.extend(get_agent_gravity(agent_pos,planets))
            new_data_dict["input"].append(new_data_input)

    with open("descriptive_naming.json", "w") as file:
        json_str = json.dumps(new_data_dict, indent=4)
        file.write(json_str)

    bar.next()
    bar.finish()
