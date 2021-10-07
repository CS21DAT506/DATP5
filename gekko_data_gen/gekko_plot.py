import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl


class GekkoPlotter:
    def __init__(self, results):

        legend_settings = {
            "loc":'upper left',
            "borderaxespad": 0,
            "bbox_to_anchor": (1.05, 1)
        }

        plot_settings = {
            "mv_out": {
                "rows": 4,
                "columns": 2,
                "number": 1
            },
            "sp": {
                "rows": 4,
                "columns": 2,
                "number": 3
            },
            "gravity": {
                "rows": 4,
                "columns": 2,
                "number": 5
            },
            "speed": {
                "rows": 4,
                "columns": 2,
                "number": 7
            }
        }

        x_min = np.min(results["time"])
        x_max = np.max(results["time"])/5

        plt.figure()

        GekkoPlotter.plotMVOutput(results, x_min, x_max, plot_settings, legend_settings)
        GekkoPlotter.plotSPGraph(results, x_min, x_max, plot_settings, legend_settings)
        GekkoPlotter.plotGravityEffect(results, x_min, x_max, plot_settings, legend_settings)
        GekkoPlotter.plotSpeed(results, x_min, x_max, plot_settings, legend_settings)
        
        plt.show()

    def vector_length(vector_x, vector_y):
        return [math.sqrt(e1**2 + e2**2) for e1, e2 in zip(vector_x, vector_y)]

    def plotMVOutput(results, xaxis_min, xaxis_max, plot_settings, legend_settings):
        plt.subplot(plot_settings["mv_out"]["rows"], plot_settings["mv_out"]["columns"], plot_settings["mv_out"]["number"])
        plt.xlim(xaxis_min, xaxis_max)
        plt.plot(results["time"],results["agent_ax"],color="blue",label="MV Optimized")
        plt.plot(results["time"],results["agent_ay"],color="red", label="MV Optimized")
        plt.legend(**legend_settings)
        plt.ylabel("Input")

    def plotSPGraph(results, xaxis_min, xaxis_max, plot_settings, legend_settings):
        plt.subplot(plot_settings["sp"]["rows"], plot_settings["sp"]["columns"], plot_settings["sp"]["number"])
        plt.xlim(xaxis_min, xaxis_max)
        plt.plot(results["time"],results["agent_px.tr"],"k-",label="Reference px Trajectory")
        plt.plot(results["time"],results["agent_px"],"g--",label="CV px Response")
        plt.plot(results["time"],results["agent_py.tr"],"b-",label="Reference py Trajectory")
        plt.plot(results["time"],results["agent_py"],"r--",label="CV py Response")
        plt.ylabel("Output")
        plt.xlabel("Time")
        plt.legend(**legend_settings)

    def plotGravityEffect(results, xaxis_min, xaxis_max, plot_settings, legend_settings):
        plt.subplot(plot_settings["gravity"]["rows"], plot_settings["gravity"]["columns"], plot_settings["gravity"]["number"])
        plt.xlim(xaxis_min, xaxis_max)
        #plt.plot(results["time"], [math.sqrt((px[i] - planetX)**2 + (py[i] - planetY)**2) for i in range(len(px))], "b-", label="Total acceleration")
        # G * massPlanet * (planetX - px) / ((planetX - px)**2 + (planetY - py)**2)**(3/2)
        gravity_x = [results["gravity_x"][i] / results["dist"][i] for i in range(len(results["gravity_x"]))]
        gravity_y = [results["gravity_y"][i] / results["dist"][i] for i in range(len(results["gravity_y"]))]
        
        plt.plot(results["time"], gravity_x, color="blue", label="gravity x")
        plt.plot(results["time"], gravity_y, color="red", label="gravity y")
        plt.plot(results["time"], GekkoPlotter.vector_length(gravity_x, gravity_y), color="green", label="gravity length")
        plt.legend(**legend_settings)

    def plotSpeed(results, xaxis_min, xaxis_max, plot_settings, legend_settings):
        plt.subplot(plot_settings["speed"]["rows"], plot_settings["speed"]["columns"], plot_settings["speed"]["number"])
        plt.xlim(xaxis_min, xaxis_max)
        plt.plot(results["time"], GekkoPlotter.vector_length(results["agent_vx"], results["agent_vy"]), color="green", label="speed length")
        plt.legend(**legend_settings)

    def plot3DGraph(results, planets, target_pos):
        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        axe = fig.gca(projection='3d')

        axe.plot(results["agent_px"], results["agent_py"], results["time"], "o", label='parametric curve')
        axe.plot([target_pos[0] for i in range(len(results["time"]))], [target_pos[1] for i in range(len(results["time"]))], results["time"], "o", label='target')
        for planet in planets:
            axe.plot([planet["initial_pos"][0] for i in range(len(results["time"]))], [planet["initial_pos"][1] for i in range(len(results["time"]))], results["time"], "o", label="planet")
        axe.legend()
        plt.show()

