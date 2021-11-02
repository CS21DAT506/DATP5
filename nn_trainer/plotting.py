import matplotlib.pyplot as plt 
import numpy as np



def plot_funcs(func_1, func_1_label, func_2, func_2_label, 
                    start=-10, stop=20,
                    x_label="x", y_label="y", 
                    title="Smoketest"):
    x = np.linspace(start, stop)
    plt.plot(x, [ func_1(inp) for inp in x], label=func_1_label)
    plt.plot(x, [ func_2(inp) for inp in x], label=func_2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_nn_funcs(func_1, func_1_label, func_2, func_2_label, 
                    start=-10, stop=20,
                    x_label="x", y_label="y", 
                    title="Smoketest"):
    x = np.linspace(start, stop)
    plt.plot(x, [ func_1(inp) for inp in x], label=func_1_label)
    plt.plot(x, [ func_2([inp])[0][0] for inp in x], label=func_2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

