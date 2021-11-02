import numpy

def parabola(x):
    return 3 * x**2 + 7 * x 

def get_data_points():
    res = numpy.linspace(-300, 300, 601)
    tf_input_list = [ [[num]] for num in res ]
    tf_output_list = [ [[parabola(num)]] for num in res ]
    return tf_input_list, tf_output_list
