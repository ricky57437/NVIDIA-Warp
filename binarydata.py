import csv
from pxr import Usd, UsdGeom
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


stage = Usd.Stage.Open("binary.usd")
#opens the usd file and assigns to stage
prim = stage.GetPrimAtPath("/root/particles")
#prims are like headers. assigns prim to the /root/particles prim
positions_attr = prim.GetAttribute("positions")
#within prims are attributes. assigns positions_attr to particle positions
velocities_attr = prim.GetAttribute("velocities")

time_samples = positions_attr.GetTimeSamples()

p0_pos = []
p0_vel = [10,]
p1_pos = []
p1_vel = [-10,]
time = []
#creating empty arrays

fps = stage.GetTimeCodesPerSecond()

for t in time_samples:
#iterating through every time block (400 of them)
    positions = positions_attr.Get(t)
    #gets particle positions at time t
    velocities = velocities_attr.Get(t)
    p0 = positions[0]
    #gets only particle 0's position
    p1 = positions[1]
    p0_pos.append(p0[0])
    #if want all 3 coordinates do (p0[0], p0[1], p0[2])
    p1_pos.append(p1[0])
    time.append(t / fps)
    #t is the time step and fps is number of t's per second


for i in range(1, len(time)):
    dt = time[i] - time[i - 1]
    v0 = (p0_pos[i] - p0_pos[i - 1]) / dt
    v1 = (p1_pos[i] - p1_pos[i - 1]) / dt
    p0_vel.append(v0)
    p1_vel.append(v1)



#PLOTTING
time = np.array(time)
p0_pos = np.array(p0_pos)
p0_vel = np.array(p0_vel)
p1_pos = np.array(p1_pos)
p1_vel = np.array(p1_vel)
#converting from py arrays to np arrays

plt.figure(figsize=(12, 6))
plt.plot(time, p0_pos, label="p0_pos vs time")
plt.plot(time, p1_pos, label="p1_pos vs time")
plt.xticks(range(1, 7))
plt.yticks([-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30])
plt.xlabel("Time")
plt.ylabel("Position (x)")
plt.title("Particle p0 and p1 Positions Over Time")
plt.legend()
plt.grid(True)
#position vs time plot

plt.figure(figsize=(12, 6))
plt.plot(time, p0_vel, label="p0_vel vs time")
plt.plot(time, p1_vel, label="p1_vel vs time")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Particle p0 and p1 Velocities Over Time")
plt.legend()
plt.grid(True)


print(len(time_samples))
print(p0_vel)
print(p0_vel[-1] / 10)
#plt.show()
