import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.integrate import odeint

def odesys(y, t, M, m, c, k, l, g):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = M + m
    a12 = m*l*np.cos(y[1])
    a21 = m*l*np.cos(y[1])
    a22 = m*l**2

    b1 = m*l*y[3]**2*np.sin(y[1]) - k*y[0]
    b2 = m*g*l*np.sin(y[1]) - c*y[1]

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy

M = 5
m = 1
c = 10
k = 20
l = 5
g = 9.81

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

x0 = 0
phi0 = 0.5
dx0 = 0
dphi0 = 0
y0 = [x0, phi0, dx0, dphi0]

Y = odeint(odesys, y0, t, (M, m, c, k, l, g))

x = Y[:,0]
phi = Y[:,1]
dx = Y[:,2]
dphi = Y[:,3]
ddx = [odesys(y,t,M,m,c,k,l,g)[2] for y,t in zip(Y,t)]
ddphi = [odesys(y,t,M,m,c,k,l,g)[3] for y,t in zip(Y,t)]

RA = m*g*np.cos(phi) - m*l*dphi**2 + m*ddx*np.sin(phi)

fig_for_graphs = plt.figure(figsize=[13,7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, x, color='Blue')
ax_for_graphs.set_title("x(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, phi, color='Red')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, RA, color='Black')
ax_for_graphs.set_title("RA(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

SprX_0 = 4
BoxX = 6
BoxY = 2
WheelR = 0.5

X_A = SprX_0 + x + BoxX/2
Y_A = 2*WheelR + BoxY/2
X_B = X_A + l*np.sin(phi)
Y_B = Y_A + l*np.cos(phi)

X_Box = np.array([-BoxX/2, BoxX/2, BoxX/2, -BoxX/2, -BoxX/2])
Y_Box = np.array([BoxY/2, BoxY/2, -BoxY/2, -BoxY/2, BoxY/2])

psi = np.linspace(0, 2*math.pi, 20)
X_Wheel = WheelR*np.sin(psi)
Y_Wheel = WheelR*np.cos(psi)

X_C1 = SprX_0 + x + BoxX/5
Y_C1 = WheelR
X_C2 = SprX_0 + x + 4*BoxX/5
Y_C2 = WheelR

X_Ground = [0, 0, 15]
Y_Ground = [7, 0, 0]

K = 19
Sh = 0.4
b = 1/(K-2)
X_Spr = np.zeros(K)
Y_Spr = np.zeros(K)
X_Spr[0] = 0
Y_Spr[0] = 0
X_Spr[K-1] = 1
Y_Spr[K-1] = 0
for i in range(K-2):
    X_Spr[i+1] = b*((i+1) - 1/2)
    Y_Spr[i+1] = Sh*(-1)**i

L_Spr = SprX_0+x

Nv = 3
R1 = 0.2
R2 = 1
thetta = np.linspace(0, Nv*2*math.pi-phi[0], 100)
X_SpiralSpr = -(R1 + thetta*(R2-R1)/thetta[-1])*np.sin(thetta)
Y_SpiralSpr = (R1 + thetta*(R2-R1)/thetta[-1])*np.cos(thetta)

alpha = x/WheelR

fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[0, 12], ylim=[-4, 10])
ax.plot(X_Ground, Y_Ground, color='black', linewidth=3)

Drawed_Box = ax.plot(X_Box+X_A[0], Y_Box+Y_A)[0]
Drawed_Wheel1 = ax.plot(X_Wheel+X_C1[0], Y_Wheel+Y_C1)[0]
Drawed_Wheel2 = ax.plot(X_Wheel+X_C2[0], Y_Wheel+Y_C2)[0]
Drawed_AB = ax.plot([X_A[0], X_B[0]], [Y_A, Y_B[0]])[0]

Point_A = ax.plot(X_A[0], Y_A, marker='o')[0]
Point_B = ax.plot(X_B[0], Y_B[0], marker='o', markersize=20)[0]

Drawed_Spring = ax.plot(X_Spr*L_Spr[0], Y_Spr+Y_A)[0]
Drawed_Spiral_Spring = ax.plot(X_SpiralSpr+X_A[0], Y_SpiralSpr+Y_A)[0]

Drawed_Wheel1_D1 = ax.plot([X_C1[0]+WheelR*np.sin(alpha[0]), X_C1[0]-WheelR*np.sin(alpha[0])],
                           [Y_C1+WheelR*np.cos(alpha[0]), Y_C1-WheelR*np.cos(alpha[0])])[0]
Drawed_Wheel1_D2 = ax.plot([X_C2[0]+WheelR*np.sin(alpha[0]), X_C2[0]-WheelR*np.sin(alpha[0])],
                           [Y_C2+WheelR*np.cos(alpha[0]), Y_C2-WheelR*np.cos(alpha[0])])[0]

def anima(i):
    Drawed_Box.set_data(X_Box+X_A[i], Y_Box+Y_A)
    Drawed_Wheel1.set_data(X_Wheel+X_C1[i], Y_Wheel+Y_C1)
    Drawed_Wheel2.set_data(X_Wheel+X_C2[i], Y_Wheel+Y_C2)
    Drawed_Spring.set_data(X_Spr*L_Spr[i], Y_Spr+Y_A)
    Drawed_AB.set_data([X_A[i], X_B[i]], [Y_A, Y_B[i]])
    Point_A.set_data(X_A[i], Y_A)
    Point_B.set_data(X_B[i], Y_B[i])
    thetta = np.linspace(0, Nv * 2 * math.pi - phi[i], 100)
    X_SpiralSpr = -(R1 + thetta * (R2 - R1) / thetta[-1]) * np.sin(thetta)
    Y_SpiralSpr = (R1 + thetta * (R2 - R1) / thetta[-1]) * np.cos(thetta)
    Drawed_Spiral_Spring.set_data(X_SpiralSpr+X_A[i], Y_SpiralSpr+Y_A)
    Drawed_Wheel1_D1.set_data([X_C1[i]+WheelR*np.sin(alpha[i]), X_C1[i]-WheelR*np.sin(alpha[i])],
                              [Y_C1+WheelR*np.cos(alpha[i]), Y_C1-WheelR*np.cos(alpha[i])])
    Drawed_Wheel1_D2.set_data([X_C2[i] + WheelR * np.sin(alpha[i]), X_C2[i] - WheelR * np.sin(alpha[i])],
                              [Y_C2 + WheelR * np.cos(alpha[i]), Y_C2 - WheelR * np.cos(alpha[i])])
    return [Drawed_Box, Drawed_Wheel1, Drawed_Wheel2, Drawed_Spring, Drawed_AB, Point_A, Point_B, Drawed_Spiral_Spring,
            Drawed_Wheel1_D1, Drawed_Wheel1_D2]

anim = FuncAnimation(fig, anima, frames=len(t), interval=40, repeat=False)

plt.show()