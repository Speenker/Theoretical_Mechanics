import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Steps = 1000
t_fin = 20
t = np.linspace(0, t_fin, Steps)

x = np.sin(t * math.pi)
phi = np.cos(2 * t) - np.sin(2 * t)

# constants for variant
SprX_0 = 2.5
R = 1
r = 0.2
L_Spr = SprX_0 + x + R

# OX and OY 
X_Ground = [0, 0, 6]
Y_Ground = [6, 0, 0]

# cylinder centers
X_O = L_Spr
Y_O = R

X_B = X_O + (R - r) * np.sin(phi)
Y_B = Y_O - (R - r) * np.cos(phi)

# arrays for cylinders
psi = np.linspace(0, 2*math.pi, 100)
X_R = R*np.sin(psi)
Y_R = R*np.cos(psi)
X_r = r*np.sin(psi)
Y_r = r*np.cos(psi)

# creating spring
K = 20 # number of points in spring
Sh = 0.1 # height of a spring line
b = 1 / (K - 2) # step
X_Spr = np.zeros(K)
Y_Spr = np.zeros(K)
X_Spr[0] = 0
Y_Spr[0] = 0
X_Spr[K - 1] = 1
Y_Spr[K - 1] = 0
for i in range(K - 2):
    X_Spr[i + 1] = b * ((i + 1) - 1/2)
    Y_Spr[i + 1] = Sh * (-1) ** i

# plotting
fig = plt.figure(figsize=[10, 7])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[0, 8], ylim=[-1, 7])

# adding all figures to plot
Drawed_R = ax.plot(X_O[0] + X_R, Y_O + Y_R)[0]
Drawed_r = ax.plot(X_B[0] + X_r, Y_B[0] + Y_r)[0]

Point_O = ax.plot(X_O[0], Y_O, marker = 'o')[0]

Drawed_Spring = ax.plot(X_Spr * L_Spr[0], Y_Spr + Y_O)[0]

ax.plot(X_Ground, Y_Ground, color='Black', linewidth=2)

# animating function
def anima(i):
    Drawed_R.set_data(X_O[i] + X_R, Y_O + Y_R)
    Drawed_r.set_data(X_B[i] + X_r, Y_B[i] + Y_r)
    Point_O.set_data(X_O[i], Y_O)
    Drawed_Spring.set_data(X_Spr * L_Spr[i], Y_Spr + Y_O)

    return [Drawed_R, Drawed_r, Point_O, Drawed_Spring]

anim = FuncAnimation(fig, anima, frames=len(t), interval=30, repeat=False)

plt.show()