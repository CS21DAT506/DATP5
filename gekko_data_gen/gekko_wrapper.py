from gekko import GEKKO
import numpy as np
import datetime

class Gekko:

    def __init__(self, remote=False):
        self.m = GEKKO(remote)
        self.m.options.MAX_ITER = 1000
        self.m.time = np.linspace(0,40,201)
        self.m.G = 1
        self.m.options.IMODE = 6  # control
        self.m.options.MV_STEP_HOR = 5  # how often the MVs can change values
        self.m.options.PRED_TIME = 100.0
        self.m.options.WEB = 0
        self.m.options.MAX_TIME = 40.0
        self.m.options.CV_TYPE = 2 # squared error

    def setup(self, agent, planets, target_pos=[100,100], acceleration_bound=100):
        planet = planets[0]

        # Manipulated variable (acceleration of agent)
        ax = self.m.MV(value=0, lb=-acceleration_bound, ub=acceleration_bound, name="agent_ax")
        ax.STATUS = 1  # allow optimizer to change
        ax.DCOST = 0.001 # smooth out gas pedal movement
        ax.DMAX = acceleration_bound/5   # slow down change of gas pedal

        ay = self.m.MV(value=0, lb=-acceleration_bound, ub=acceleration_bound, name="agent_ay")
        ay.STATUS = 1
        ay.DCOST = 0.001
        ay.DMAX = acceleration_bound/5

        # Controlled Variable (position of agent)
        px = self.m.CV(value=agent["initial_pos"][0], name="agent_px")
        px.STATUS = 1
        px.SP = target_pos[0]
        px.SPHI = target_pos[0] + 5
        px.SPLO = target_pos[0] - 5
        px.TR_INIT = 0

        py = self.m.CV(value=agent["initial_pos"][1], name="agent_py")
        py.STATUS = 1
        py.SP = target_pos[1]
        py.SPHI = target_pos[1] + 5
        py.SPLO = target_pos[1] - 5
        py.TR_INIT = 0

        # Other model variables
        vx = self.m.Var(value=agent["initial_velocity"][0], name="agent_vx")
        vy = self.m.Var(value=agent["initial_velocity"][1], name="agent_vy")

        planet_vx = self.m.Const(value=planet["initial_velocity"][0], name="planet_vx")
        planet_vy = self.m.Const(value=planet["initial_velocity"][1], name="planet_vy")

        planet_x = self.m.Var(value=planet["initial_pos"][0], name="planet_px")
        planet_y = self.m.Var(value=planet["initial_pos"][1], name="planet_py")

        # Planet position calculations
        

        # Intermediate calculations
        gx = self.m.Intermediate(self.m.G * planet["mass"] * (planet_x - px), name="gravity_x")
        gy = self.m.Intermediate(self.m.G * planet["mass"] * (planet_y - py), name="gravity_y")

        dist = self.m.Intermediate(((planet_x - px)**2 + (planet_y - py)**2)**(3/2), name="dist")

        # Process model
        self.m.Equation([
            #gx == G * massPlanet * (planetX - px) / ((planetX - px)**2 + (planetY - py)**2)**(3/2),
            #gy == G * massPlanet * (planetY - py) / ((planetX - px)**2 + (planetY - py)**2)**(3/2),
            planet_x.dt() == planet_vx,
            planet_y.dt() == planet_vy,
            vx.dt() * dist == ax * dist + gx, 
            vy.dt() * dist == ay * dist + gy, 
            px.dt() == vx, 
            py.dt() == vy,
            #planet_vx.dt() == 0,
            #planet_vy.dt() == 0,

            ax**2 + ay**2 < acceleration_bound**2, 
            ((px - planet_x)**2 + (py - planet_y)**2) > planet["radius"]**2
        ])

        self.m.Maximize(dist)
        self.m.Obj((ax**2 + ay**2) * self.m.time[-1] / len(self.m.time))

    def solve(self, *args, **kwargs):
        with open(f"{datetime.datetime.now().strftime('%Y_%m_%d')}.txt", "a") as f:
            f.write(self.m.path + "\n")

        self.m.solve(*args, **kwargs)

