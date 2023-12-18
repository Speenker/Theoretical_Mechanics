import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constant for decreasing the size of the arrows
decrease = 7

# Rotate function
def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY

T = np.linspace(0, 10, 1000)
t = sp.Symbol('t')

# Given values
phi = 5 * t
r = 1 - sp.sin(t)


x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t) / decrease
Vy = sp.diff(y, t) / decrease
Ax = sp.diff(Vx, t) / decrease
Ay = sp.diff(Vy, t) / decrease

# Filling arrays with zeros
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)

# Filing arrays with substitute values
for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])

# Setting a plot
fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-3.5, 3.5], ylim=[-3.5, 3.5])

ax.plot(X, Y)

# Setting displayed values
point, = ax.plot(X[0], Y[0], marker = 'o')
point_pos, = ax.plot(0, 0, marker = 'o')
rad_vec, = ax.plot([0, X[0]], [0, Y[0]], 'r')
vel_vec, = ax.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'g')
acc_vec, = ax.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'b')

x_arrow = np.array([-0.2, 0, -0.2])
y_arrow = np.array([0.1, 0, -0.1])

rot_x_arrow, rot_y_arrow = Rot2D(x_arrow, y_arrow, math.atan2(VY[0], VX[0]))
rot_acc_x_arrow, rot_acc_y_arrow = Rot2D(x_arrow, y_arrow, math.atan2(AY[0], AX[0]))
rot_vel_x_arrow, rot_vel_y_arrow = Rot2D(x_arrow, y_arrow, math.atan2(Y[0], X[0]))

vel_arrow, = ax.plot(rot_x_arrow + X[0] + VX[0], rot_y_arrow + Y[0] + VY[0], 'g')
acc_arrow, = ax.plot(rot_acc_x_arrow + X[0] + AX[0], rot_acc_y_arrow + Y[0] + AY[0], 'b')
rad_arrow, = ax.plot(rot_vel_x_arrow + X[0], rot_vel_y_arrow + Y[0], 'r')

# Function for animation, that changing all values via i
def anima(i):
    point_pos.set_data(0,0)

    point.set_data(X[i], Y[i])

    rad_vec.set_data([0, X[i]], [0, Y[i]])

    vel_vec.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])

    acc_vec.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])

    rot_x_arrow, rot_y_arrow = Rot2D(x_arrow, y_arrow,math.atan2(VY[i], VX[i]))
    vel_arrow.set_data(rot_x_arrow + X[i] + VX[i], rot_y_arrow + Y[i] + VY[i])

    rot_acc_x_arrow, rot_acc_y_arrow = Rot2D(x_arrow, y_arrow,math.atan2(AY[i], AX[i]))
    acc_arrow.set_data(rot_acc_x_arrow + X[i] + AX[i], rot_acc_y_arrow + Y[i] + AY[i])

    rot_vel_x_arrow, rot_vel_y_arrow = Rot2D(x_arrow, y_arrow, math.atan2(Y[i], X[i]))
    rad_arrow.set_data(rot_vel_x_arrow + X[i], rot_vel_y_arrow + Y[i])

    return point ,vel_vec ,vel_arrow ,acc_vec ,acc_arrow, rad_vec, rad_arrow

anim = FuncAnimation(fig, anima, frames = 1000, interval = 10, repeat = False)

plt.show()