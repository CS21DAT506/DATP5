from gekko import GEKKO
import numpy as np
import datetime

class Gekko:

    def __init__(self, remote=False):
        self.m = GEKKO(remote)
        self.m.options.MAX_ITER = 100000
        self.m.time = np.linspace(0,60,401)
        self.m.G = 1
        self.m.precision = 5
        self.m.options.IMODE = 6  # control


    def setup(self, agent, planets, target_pos=[100,100], acceleration_bound=100):
        planet = planets[0]

        # Manipulated variable
        ax = self.m.MV(value=0, lb=-acceleration_bound, ub=acceleration_bound, name="agent_ax")
        ax.STATUS = 1  # allow optimizer to change
        ax.DCOST = 1 # smooth out gas pedal movement
        ax.DMAX = acceleration_bound/5   # slow down change of gas pedal

        ay = self.m.MV(value=0, lb=-acceleration_bound, ub=acceleration_bound, name="agent_ay")
        ay.STATUS = 1
        ay.DCOST = 1
        ay.DMAX = acceleration_bound/5

        # Controlled Variable
        self.m.options.CV_TYPE = 2 # squared error
        vx = self.m.Var(value=agent["initial_velocity"][0], name="agent_vx")
        vy = self.m.Var(value=agent["initial_velocity"][1], name="agent_vy")


        # Position
        px = self.m.CV(value=agent["initial_pos"][0], name="agent_px")
        px.STATUS = 1
        px.SP = target_pos[0]
        px.TR_INIT = 0

        py = self.m.CV(value=agent["initial_pos"][1], name="agent_py")
        py.STATUS = 1
        py.SP = target_pos[1]
        py.TR_INIT = 0

        gx = self.m.Intermediate(self.m.G * planet["mass"] * (planet["initial_pos"][0] - px), name="gravity_x")
        gy = self.m.Intermediate(self.m.G * planet["mass"] * (planet["initial_pos"][1] - py), name="gravity_y")

        dist = self.m.Intermediate(((planet["initial_pos"][0] - px)**2 + (
            planet["initial_pos"][1] - py)**2)**(3/2), name="dist")

        # Process model
        self.m.Equation([
            #gx == G * massPlanet * (planetX - px) / ((planetX - px)**2 + (planetY - py)**2)**(3/2),
            #gy == G * massPlanet * (planetY - py) / ((planetX - px)**2 + (planetY - py)**2)**(3/2),
            vx.dt() * dist == ax * dist + gx, 
            vy.dt() * dist == ay * dist + gy, 
            px.dt() == vx, 
            py.dt() == vy,
            ax**2 + ay**2 < acceleration_bound**2, 
            ((px - planet["initial_pos"][0])**2 + (py - planet["initial_pos"][1])**2) > planet["radius"]**2
        ])

    def solve(self, *args, **kwargs):
        with open(f"{datetime.datetime.now().strftime('%Y_%m_%d')}.txt", "a") as f:
            f.write(self.m.path + "\n")

        self.m.solve(*args, **kwargs)

