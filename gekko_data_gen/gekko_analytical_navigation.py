from gekko import GEKKO
import numpy as np

class AnalyticalNavigator:
    def __init__(self, remote=False):
        self.m = GEKKO(remote)
        self.m.time = np.linspace(0, 60, 401)
        self.m.options.IMODE = 4
    
    def setup(self, agent, planets, target_pos=[100,100], acceleration_bound=100):
        planet = planets[0]
        self.m.G = 1
        # Manipulated variable
        #ax = self.m.Var(value=0, lb=-acceleration_bound, ub=acceleration_bound, name="agent_ax")
        #ax.DMAX = acceleration_bound/5   # slow down change of gas pedal

        #ay = self.m.Var(value=0, lb=-acceleration_bound, ub=acceleration_bound, name="agent_ay")
        #ay.DMAX = acceleration_bound/5

        # Controlled Variable
        vx = self.m.Var(value=agent["initial_velocity"][0], name="agent_vx")
        vy = self.m.Var(value=agent["initial_velocity"][1], name="agent_vy")


        # Position
        px = self.m.Var(value=agent["initial_pos"][0], name="agent_px")
        py = self.m.Var(value=agent["initial_pos"][1], name="agent_py")

        extra_radius = self.m.Var(value=0, lb=0, name="extra_radius")

        gx = self.m.Intermediate(self.m.G * planet["mass"] * (planet["initial_pos"][0] - px), name="gravity_x")
        gy = self.m.Intermediate(self.m.G * planet["mass"] * (planet["initial_pos"][1] - py), name="gravity_y")

        dist = self.m.Intermediate(((planet["initial_pos"][0] - px)**2 + (planet["initial_pos"][1] - py)**2)**(3/2), name="dist")
        dist_target = self.m.Intermediate(((target_pos[0] - px)**2 + (target_pos[1] - py)**2)**(1/2), name="dist_target")

        norm_target_x = self.m.Intermediate((target_pos[0] - px) / dist_target, name="norm_target_x")
        norm_target_y = self.m.Intermediate((target_pos[1] - py) / dist_target, name="norm_target_y")
   
        ag_length = self.m.Intermediate((gx**2 + gy**2)**(1/2) / dist, name="ag_length")

        dot_product = self.m.Intermediate(norm_target_x * gx / dist + norm_target_y * gy / dist, name="dot_product")
        factor = self.m.Intermediate((-dot_product + self.m.sqrt((dot_product)**2 + ag_length**2 - acceleration_bound**2)), name="factor")

        ax = self.m.Intermediate((factor * norm_target_x - gx), name="agent_ax")
        ay = self.m.Intermediate((factor * norm_target_y - gy), name="agent_ay")

        # Process model
        self.m.Equation([
            vx.dt() * dist == ax * dist + gx, 
            vy.dt() * dist == ay * dist + gy, 
            px.dt() == vx, 
            py.dt() == vy,
            ((px - planet["initial_pos"][0])**2 + (py - planet["initial_pos"][1])**2) == planet["radius"]**2 + extra_radius,
        ])
        
    def solve(self, *args, **kwargs):
        # with open(f"{datetime.datetime.now().strftime('%Y_%m_%d')}.txt", "a") as f:
        #     f.write(self.m.path + "\n")

        self.m.solve(*args, **kwargs)
        