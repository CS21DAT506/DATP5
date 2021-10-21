import numpy as np
import json

def validate_nn_input_output(nn_input, nn_input_size):
    """
    nn_input is the input to the nn as a list/np.array
    nn_input_size is a the size of the input layer
    """

    nn_input = np.array(nn_input)
    nn_input_shape = (len(nn_input), 1, nn_input_size) # len(nn_input) refers to amount of data points

    if (nn_input.shape != nn_input_shape):
        raise Exception(f"invalid input: dimension of array:\nactual   {nn_input.shape}\nexpected {nn_input_shape}")

    return True

def load_nn_data(json_path, nn_input_size, nn_output_size):
    data = None
    with open(json_path) as file:
        data = json.load(file)

    if validate_nn_input_output(data["input"], nn_input_size) and validate_nn_input_output(data["output"], nn_output_size):
        return data["input"], data["output"]
    else:
        raise Exception("SHOULD NOT HAPPEND!")
    
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
