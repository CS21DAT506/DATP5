from typing import Tuple
from gekko import GEKKO
import numpy as np
import datetime

def calculate_planet_position(planets, time):
    acc_x, acc_y = 0, 0
    positions_time = []
    velocities_time = []
    for p in planets:
        positions_time.append([(p["initial_pos"][0], p["initial_pos"][1])])
        velocities_time.append([(p["initial_velocity"][0], p["initial_velocity"][1])])

    for i in range(1, len(time)):
        for p in range(len(planets)):
            acc_x, acc_y = 0, 0
            for pl in range(len(planets)):
                if p == pl:
                    continue
                dist = ((positions_time[pl][i - 1][0] - positions_time[p][i - 1][0])**2 + (positions_time[pl][i - 1][1] - positions_time[p][i - 1][1])**2)
                acc_x += 1 * planets[pl]["mass"] * (positions_time[pl][i - 1][0] - positions_time[p][i - 1][0]) / (dist**3) 
                acc_y += 1 * planets[pl]["mass"] * (positions_time[pl][i - 1][1] - positions_time[p][i - 1][1]) / (dist**3) 
            time_elapsed = time[i] - time[i - 1]
            curr_vel_x = velocities_time[p][i - 1][0] + acc_x * time_elapsed
            curr_vel_y = velocities_time[p][i - 1][1] + acc_y * time_elapsed
            velocities_time[p].append((curr_vel_x, curr_vel_y))
            new_x = positions_time[p][i - 1][0] + velocities_time[p][i][0] * time_elapsed
            new_y = positions_time[p][i - 1][1] + velocities_time[p][i][1] * time_elapsed
            positions_time[p].append((new_x, new_y))

    return positions_time

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
        #self.m.options.MAX_TIME = 40.0
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

        planets_px = []
        planets_py = []
        for i in range(len(planets)):
            planets_px.append(self.m.Param(value=[x for x, _ in calculate_planet_position(planets, self.m.time)[i]] ,name=f"planet{i}_px"))
            planets_py.append(self.m.Param(value=[y for _, y in calculate_planet_position(planets, self.m.time)[i]], name=f"planet{i}_py"))
        
        # Planet position calculations
        planet_grav_x = []
        planet_grav_y = []
        for i in range(len(planets)):
            planet_grav_x.append(self.m.Param(self.m.G * planets[i]["mass"] * (planets_px[i] - px) / ((planets_px[i] - px)**2 + (planets_py[i] - py)**2)**(3/2), name=f"planet{i}_gx"))
            planet_grav_y.append(self.m.Param(self.m.G * planets[i]["mass"] * (planets_py[i] - py) / ((planets_px[i] - px)**2 + (planets_py[i] - py)**2)**(3/2), name=f"planet{i}_gy"))

        gx = self.m.Intermediate(self.m.sum(planet_grav_x), name="gravity_x")
        gy = self.m.Intermediate(self.m.sum(planet_grav_y), name="gravity_y")

        # Intermediate calculations
        # gx = self.m.Intermediate(self.m.G * planet["mass"] * (planet_x - px), name="gravity_x")
        # gy = self.m.Intermediate(self.m.G * planet["mass"] * (planet_y - py), name="gravity_y")

        # dist = self.m.Intermediate(((planet_x - px)**2 + (planet_y - py)**2)**(3/2), name="dist")

        # Process model
        self.m.Equation([
            #gx == G * massPlanet * (planetX - px) / ((planetX - px)**2 + (planetY - py)**2)**(3/2),
            #gy == G * massPlanet * (planetY - py) / ((planetX - px)**2 + (planetY - py)**2)**(3/2),
            vx.dt() == ax + gx, 
            vy.dt() == ay + gy, 
            px.dt() == vx, 
            py.dt() == vy,
            #planet_vx.dt() == 0,
            #planet_vy.dt() == 0,

            ax**2 + ay**2 < acceleration_bound**2, 
        ])

        for i in range(len(planets)):
            self.m.Equation((px - planets_px[i])**2 + (py - planets_py[i])**2 > planets[i]["radius"])
            #self.m.Maximize((px - planets_px[i])**2 + (py - planets_py[i])**2)
        #self.m.Obj((ax**2 + ay**2) * self.m.time[-1] / len(self.m.time))

    def solve(self, *args, **kwargs):
        with open(f"{datetime.datetime.now().strftime('%Y_%m_%d')}.txt", "a") as f:
            f.write(self.m.path + "\n")

        self.m.solve(*args, **kwargs)


if __name__ == "__main__":
    planet = [{
        "mass": 1000000,
        "initial_pos": np.array([320,420]),
        "radius": 20,
        "initial_velocity": np.array([10,5]),
    }]

    time = np.linspace(0, 40, 201)

    print(calculate_planet_position(planet, time))
