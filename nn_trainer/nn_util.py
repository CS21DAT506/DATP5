import json

def load_nn_data(json_path):
    data = None
    with open(json_path) as file:
        data = json.load(file)
    return data["input"], data["output"]
