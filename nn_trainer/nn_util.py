import numpy as np
import json
from pathlib import Path

def ensure_valid_input_size(nn_input, expected_input_size):
    invalid_entry_count = 0
    # print(f"Input size: {len(nn_input)}")
    for entry in nn_input:
        data_point = entry[0]
        if len(data_point) != expected_input_size:
            invalid_entry_count += 1
            planet_pos_vel = data_point[-5:-1]
            for num in planet_pos_vel:
                data_point.append(num)
            data_point.append(0.0)
    # print(f"Corrected {invalid_entry_count} entries.")

def validate_nn_input_output(nn_input, nn_input_size):
    """
    nn_input is the input to the nn as a list/np.array
    nn_input_size is a the size of the input layer
    """

    for entry in nn_input:
        data_point = entry[0]
        if len(data_point) != nn_input_size:
            raise Exception(f"Invalid datapoint with length {len(data_point)}. Data point is: {data_point}. ")

        for num in data_point:
            if (type(num)) is not float:
                raise TypeError("Data point elements need to be of type float!")

    nn_input = np.array(nn_input)
    nn_input_shape = (len(nn_input), 1, nn_input_size) # len(nn_input) refers to amount of data points

    if (nn_input.shape != nn_input_shape):
        raise Exception(f"invalid input: dimension of array:\nactual   {nn_input.shape}\nexpected {nn_input_shape}")

    return True

def load_nn_data(json_path, nn_input_size, nn_output_size):
    data = None
    with open(json_path) as file:
        data = json.load(file)
    # data["input"] = [ [[element]] for element in data["input"][0] ]

    ensure_valid_input_size(data["input"], nn_input_size)

    if validate_nn_input_output(data["input"], nn_input_size) and validate_nn_input_output(data["output"], nn_output_size):
        return data["input"], data["output"]
    else:
        raise Exception("SHOULD NOT HAPPEND!")
    # return data["input"], data["output"]

if __name__ == "__main__":
    nn_input, nn_output = load_nn_data("data.json", 17, 2)
    
    test_array = [[[1,2,3,4,5,6,7]]]
    is_valid = validate_nn_input_output(test_array, 7)
    print(is_valid)
    
    test_array = [[[1,2,3,4,5,6,7]],[[1]]]
    is_valid = validate_nn_input_output(test_array, 7)
    print(is_valid)

    test_array = [[1,2,3,4,5,6,7]]
    is_valid = validate_nn_input_output(test_array, 7)
    print(is_valid)
