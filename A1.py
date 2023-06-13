# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def f(x):
    f=np.power(x,3)-3*np.power(x,2)
    return f

def fp(x):
    fp=3*np.power(x,2)-6*x
    return fp

def back_propa(b,x):
    global nx
    px=nx
    nx=px-b*(fp(nx)-f(px)/nx-px)
    print(nx)
    return nx

def animate(i):
    global b,nx
    temp=back_propa(b, i)
    x=[temp]
    y=[f(x)]
    dot.set_data(x,y)
    
nx=np.random.uniform(0.1,5)
b=float(input("Give b in range [0.01-0.1] : "))

fig, ax = plt.subplots()

x=np.arange(-1,5,0.1)
y1=f(x)
y2=fp(x)
ax.set_title("Γράφημα Gradient Descent f(x) και fp(x)")

ax.set_xlim(-1,4)
ax.set_ylim(-5,25)

ax.set_xlabel('f(x)=$x^2-3x^2$ , fp(x)=$3x^2-6x$')

ax.axvline(x=0,c="k",linewidth=0.5)
ax.text(3.5,0.2,'xaxis')

ax.axhline(y=0,c="k",linewidth=0.5)
ax.text(0.01,22.5,'yaxis')

ax.plot(x,y1,"m",linewidth=0.6)
ax.text(3.1,17,'fp(x)')

ax.plot(x,y2,"b",linewidth=0.6)
ax.text(3.1,5,'f(x)')

dot,=plt.plot([],[],"ro")




ani = FuncAnimation(fig, animate, frames=120, interval=500, repeat=False)
plt.show()