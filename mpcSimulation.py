import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.animation as animation
from gekko import GEKKO
import numpy as np
import math
import matplotlib.pyplot as plt  
#from collections import deque

m = GEKKO(remote=False)
m.options.MAX_ITER = 100000
m.time = np.linspace(0,60,401)

# Parameters
mass = 500
massPlanet = 100000
planetR = 5
G = 1
planetX = 30
planetY = 20
defaultVal = 1e-10

# Manipulated variable
ax = m.MV(value=0, lb=-100, ub=100)
ax.STATUS = 1  # allow optimizer to change
ax.DCOST = 1 # smooth out gas pedal movement
ax.DMAX = 20   # slow down change of gas pedal

ay = m.MV(value=0, lb=-100, ub=100)
ay.STATUS = 1
ay.DCOST = 1
ay.DMAX = 20

# Controlled Variable
#v = m.CV(value=0)
#v.STATUS = 0  # add the SP to the objective
m.options.CV_TYPE = 2 # squared error
#v.SP = 40     # set point
#v.TR_INIT = 0 # set point trajectory
#v.TAU = 2     # time constant of trajectory
vx = m.Var(value=-10)
vy = m.Var(value=0)


# Position
px = m.CV(value=0)
px.STATUS = 1
#px.SPHI = 105
px.SP = 100
#px.SPLO = 95
px.TR_INIT = 0
#px.TR_OPEN = 1
#px.TAU = 5

py = m.CV(value=0)
py.STATUS = 1
#py.SPHI = 105
py.SP = 100
#py.SPLO = 95
py.TR_INIT = 0
#py.TR_OPEN = 1
#py.TAU = 5

gx = m.Intermediate(G * massPlanet * (planetX - px))
gy = m.Intermediate(G * massPlanet * (planetY - py))

dist = m.Intermediate(((planetX - px)**2 + (planetY - py)**2)**(3/2))

# Process model
m.Equation([
    #gx == G * massPlanet * (planetX - px) / ((planetX - px)**2 + (planetY - py)**2)**(3/2),
    #gy == G * massPlanet * (planetY - py) / ((planetX - px)**2 + (planetY - py)**2)**(3/2),
    vx.dt() * dist == ax * dist + gx, 
    vy.dt() * dist == ay * dist + gy, 
    px.dt() == vx, 
    py.dt() == vy,
    ax**2 + ay**2 < 100**2, 
    ((px - planetX)**2 + (py - planetY)**2) > planetR**2
])

m.options.IMODE = 6 # control
print(m._objectives)
m.solve(disp=False)

# get additional solution information
import json
with open(m.path+'//results.json') as f:
    results = json.load(f)
print(results.keys())
def plot2DGraphs():
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(m.time,ax.value,color="blue",label='MV Optimized')
    plt.plot(m.time,ay.value,color="red", label='MV Optimized')
    plt.legend()
    plt.ylabel('Input')
    plt.subplot(3,1,2)
    plt.plot(m.time,results['v3.tr'],'k-',label='Reference Trajectory')
    plt.plot(m.time,px.value,'r--',label='CV Response')
    plt.plot(m.time,results['v4.tr'],'b-',label='Reference Trajectory')
    plt.plot(m.time,py.value,'r--',label='CV Response')
    plt.ylabel('Output')
    plt.xlabel('Time')
    plt.legend()
    plt.subplot(3,1,3)
    #plt.plot(m.time, [math.sqrt(ax.value[i]**2 + ay.value[i]**2) for i in range(len(ax.value))], 'b-', label='Total acceleration')
    #plt.plot(m.time, [math.sqrt((px.value[i] - planetX)**2 + (py.value[i] - planetY)**2) for i in range(len(px.value))], 'b-', label='Total acceleration')
    # G * massPlanet * (planetX - px) / ((planetX - px)**2 + (planetY - py)**2)**(3/2)
    plt.plot(m.time, [gx.value[i] / dist.value[i] for i in range(len(gx.value))], color="blue", label='Total acceleration')
    plt.plot(m.time, [gy.value[i] / dist.value[i] for i in range(len(gy.value))], color="red", label='Total acceleration')


plot2DGraphs()




def plot3DGraph():
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    axe = fig.gca(projection='3d')

    axe.plot(px.value, py.value, m.time, "o", label='parametric curve')
    axe.plot([planetX for i in range(len(m.time))], [planetY for i in range(len(m.time))], m.time, "o", label="planet")
    axe.legend()

plot3DGraph()

plt.show()

"""
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], ',-', lw=1)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=len(ax.value)), deque(maxlen=len(ay.value))


def animate(i):
    thisx = [0, ax.value[i], ax.value[i+1]]
    thisy = [0, ay.value[i], ay.value[i+1]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*m.dt))
    return line, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(ay.value), interval=m.dt*1000, blit=True)
plt.show()"""