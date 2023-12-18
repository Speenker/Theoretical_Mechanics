import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# building system of diff equasions
def odesys(y, t, F0, M, m, c, gamma, g):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = m*(R-r)*(1+np.cos(y[0]))
    a12 = 2*R*(M+m)
    a21 = 2*(R-r)
    a22 = R*(1+np.cos(y[0]))

    b1 = F0*np.sin(t*gamma) - c*R*y[1] + m*(R-r)*y[2]**2*np.sin(y[0])
    b2 = -g*np.sin(y[0])

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy

# constants
F0 = 3
M = 5
m = 1
c = 10
gamma = math.pi
g = 9.81
R = 1
r = 0.3

Steps = 1000
t_fin = 20
t = np.linspace(0, t_fin, Steps)

x0 = math.pi / 2
phi0 = 0
dx0 = 0
dphi0 = 0
y0 = [x0, phi0, dx0, dphi0]

# solving the system
Y = odeint(odesys, y0, t, (F0, M, m, c, gamma, g))

x = Y[:,0]
phi = Y[:,1]
dx = Y[:,2]
dphi = Y[:,3]
ddx = [odesys(y,t,F0,M,m,c,gamma,g)[2] for y,t in zip(Y,t)]
ddphi = [odesys(y,t,F0,M,m,c,gamma,g)[3] for y,t in zip(Y,t)][0]

FA = (M+m)*R*ddphi + m*(R-r)*(ddx*np.cos(x)-dx**2*np.sin(x)) + c*R*phi - F0*np.sin(t*gamma)
NA = (M+m)*g + m*(R-r)*(ddx*np.sin(x)+dx**2*np.cos(x))

# figure for graphs
fig_for_graphs = plt.figure(figsize=[13,7])

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, x, color='Blue')
ax_for_graphs.set_title("psi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, phi, color='Red')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, FA, color='Orange')
ax_for_graphs.set_title("FA(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(t, NA, color='Black')
ax_for_graphs.set_title("NA(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

# spring constants
SprX_0 = 2.5
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