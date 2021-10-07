import matplotlib.pyplot as plt
import numpy as np
import math


class GekkoPlotter:
    def __init__(self, results):

        legend_settings = {
            "loc":'upper left',
            "borderaxespad": 0,
            "bbox_to_anchor": (1.05, 1)
        }

        x_min = np.min(results["time"])
        x_max = np.max(results["time"])/10


        plt.figure()

        plt.subplot(4,2,1)
        plt.xlim(x_min, x_max)
        plt.plot(results["time"],results["agent_ax"],color="blue",label="MV Optimized")
        plt.plot(results["time"],results["agent_ay"],color="red", label="MV Optimized")
        plt.legend(**legend_settings)
        plt.ylabel("Input")

        plt.subplot(4,2,3)
        plt.xlim(x_min, x_max)
        plt.plot(results["time"],results["agent_px.tr"],"k-",label="Reference px Trajectory")
        plt.plot(results["time"],results["agent_px"],"g--",label="CV px Response")
        plt.plot(results["time"],results["agent_py.tr"],"b-",label="Reference py Trajectory")
        plt.plot(results["time"],results["agent_py"],"r--",label="CV py Response")
        plt.ylabel("Output")
        plt.xlabel("Time")
        plt.legend(**legend_settings)
        
        plt.subplot(4,2,5)
        plt.xlim(x_min, x_max)
        #plt.plot(results["time"], [math.sqrt((px[i] - planetX)**2 + (py[i] - planetY)**2) for i in range(len(px))], "b-", label="Total acceleration")
        # G * massPlanet * (planetX - px) / ((planetX - px)**2 + (planetY - py)**2)**(3/2)
        gravity_x = [results["gravity_x"][i] / results["dist"][i] for i in range(len(results["gravity_x"]))]
        gravity_y = [results["gravity_y"][i] / results["dist"][i] for i in range(len(results["gravity_y"]))]
        
        plt.plot(results["time"], gravity_x, color="blue", label="gravity x")
        plt.plot(results["time"], gravity_y, color="red", label="gravity y")
        plt.plot(results["time"], self.vector_length(gravity_x, gravity_y), color="green", label="gravity length")
        plt.legend(**legend_settings)

        plt.subplot(4, 2, 7)
        plt.xlim(x_min, x_max)
        plt.plot(results["time"], self.vector_length(results["agent_vx"], results["agent_vy"]), color="green", label="gravity length")
        plt.legend(**legend_settings)

        plt.show()

    def vector_length(self, vector_x, vector_y):
        return [math.sqrt(e1**2 + e2**2) for e1, e2 in zip(vector_x, vector_y)]

