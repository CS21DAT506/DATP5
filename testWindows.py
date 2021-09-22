import rebound
import random
import math

#sim = rebound.SimulationArchive("archive.bin")[0]

def get_spherical(max_radius) :
    r = math.pow(random.random(), 1/3) * max_radius
    a = ( 2 * random.random() * math.pi)
    h = (1 - 2 * random.random())
    p = (math.sqrt(1 - h * h))
    x = math.cos(a) * p * r
    y = math.sin(a) * p * r
    z = h * r
    return x, y, z

def add_particle(sim):
    pos = get_spherical(1)
    vel = get_spherical(0.001)
    mass = random.random() + 0.5
    radius = mass / 10000
    sim.add(m = mass, r = radius, x = pos[0], y = pos[1], z = pos[2], vx = vel[0], vy = vel[1], vz = vel[2])

def setup():
    sim = rebound.Simulation()
    sim.integrator = "SABA(10,6,4)"
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    sim.dt = 0.001
    sim.units = ("kg", "km", "yr")
    for i in range(10):
        add_particle(sim)
    return sim

sim = setup()

print(len(sim.particles))

#sim.integrate(100)

for p in sim.particles:
   print(str(p.x), str(p.y), str(p.z), str(p.m))

