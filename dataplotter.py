import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

#creating python arrays for pos and time
#y and z shouldnt matter right now (1D)
time = []
x1 = []
y1 = []
z1 = []

x2 = []

#collecting the data from csv file
with open("particle_positions.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        time.append(float(row["time"]))
        x1.append(float(row["x1"]))
        y1.append(float(row["y1"]))
        z1.append(float(row["z1"]))

        x2.append(float(row["x2"]))

#converting from python arrays to np arrays
time = np.array(time)
x1 = np.array(x1)
y1 = np.array(y1)
z1 = np.array(z1)

x2 = np.array(x2)

#velocity in x direction of p1
vx1 = np.diff(x1) / np.diff(time)
#matches vx1 length to "time" array length
vx1 = np.insert(vx1, 0, vx1[0])


plt.plot(time, x1, label = "x1 vs time")
plt.plot(time, x2, label = "x2 vs time")
plt.xticks(range(1,8))
plt.yticks(range(1,11))
plt.xlabel("Time")
plt.ylabel("x")
plt.legend()
plt.title("Particle x1 and x2 Positions Over Time")

plt.show()



