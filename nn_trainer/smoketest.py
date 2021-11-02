import numpy

def parabola(x):
    return x**2 

def sinus(x):
    return numpy.sin(x)

def linear(x):
    return 2 * x

def get_smoketest_data_points(func=parabola, start=-300, stop=300, num=601):
    res = numpy.linspace(start, stop, num)
    tf_input_list = [ [num] for num in res ]
    tf_output_list = [ [func(num)] for num in res ]
    return tf_input_list, tf_output_list

def generate_unseen_data_points(func=parabola, start=-300.5, stop=330.5, num=30):
    res = numpy.linspace(start, stop, num)
    tf_input_list = [ [num] for num in res ]
    tf_output_list = [ [func(num)] for num in res ]
    return tf_input_list, tf_output_list

