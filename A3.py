# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x,y):
    sigma=np.sqrt(2)
    return np.sqrt(np.power(x,2)+np.power(y,2)+sigma)

def gradient(x, y):
    return np.array([x/f(x,y), y/f(x,y)])

def animate(i):
    global x, y, z
    grad = gradient(x, y)
    x = x - b * grad[0]
    y = y - b * grad[1]
    z = f(x, y)
    dot.set_data([x], [y])
    dot.set_3d_properties([z],"z")
    print("x: ",x," y: ",y)

x = 4
y = 4
z = f(x, y)

b=float(input("Give b in range [0.1-1] : "))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = np.linspace(-4, 4, 100)
y_vals = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_vals, y_vals)
z_vals = f(X, Y)

ax.plot_surface(X,Y,z_vals,cmap="plasma")

ax.set_title('$f(x,y)=sqrt(x^2+y^2+sqrt(2))$')
ax.set_xlim3d([-4, 4])
ax.set_xlabel('X')
ax.set_ylim3d([-4, 4])
ax.set_ylabel('Y')
ax.set_zlim3d([1, 6])
ax.set_zlabel('Z')

dot,=ax.plot([x],[y],[z],"ko",zorder=5)

plt.show()

ani = FuncAnimation(fig, animate, frames=120, interval=500)