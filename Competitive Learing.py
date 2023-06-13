# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
epoch=0

def animate(i):
    global epoch,syek,geit
    while epoch<=200 and syek>0.001:
        for i in range(n):
            apostash=np.zeros(neyrones)
            for j in range(neyrones):
                apostash[j] = np.linalg.norm(data[i, 0:2] - w[j, 0:2])
            thesi = np.argmin(apostash)
            w[thesi,:]=w[thesi,:]+syek*(data[i,:2]-w[thesi,:2])
            plt.figure(1)
            plt.cla()
            plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
            plt.plot(w[:neyrones, 0], w[:neyrones, 1], 'ko', markersize=12, markerfacecolor='k')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xlabel('Axis x')
            plt.ylabel('Axis y')
            plt.title(f'Protypa-Epoch {epoch}-Synapsewn {syek}')
            plt.pause(0.005)
        syek=syek*(1-epoch/200)
        epoch=epoch+1
    
        
#arxh
fig=plt.figure()
plt.subplots_adjust(wspace=0.5,hspace=0.5)
n=int(input('\n Plhthos apo dedomena(pollaplasio toy 4 ): '))
neyrones=int(input('\n Dwse arithmo antagwnistikwn neyrwnwn: '))
syek=float(input('\n bhma(0.1): '))
geit=1
w = np.random.rand(neyrones, 2)
        
data=np.zeros((n,2))
choice=1
while choice!=0:
    print("Epeleje ena apo ta parakatw: ")
    print("\n1.Grammika Diaxwrisima Protupa\n2.Mh Grammika Diaxwrisima Protupa (Klash 0 sth Gwnia)\n3.Mh Grammika Diaxwrisima Protupa (Klash 0 sto Kentro)\n4.Mh Grammika Diaxwrisima Protupa (XOR)\n5.Mh Grammika Diaxwrisima Protupa (Klash 0 mesa sthn Klash 1)\n0.Exit\n")
    choice=int(input("Epilogh: "))
    while choice<0 or choice>5:
        choice=input("Epilogh")
    if choice==1:
        data[:n//2, :2] = 0.4 * np.random.rand(n//2, 2)
        data[n//2:n, :2] = 0.5 * np.random.rand(n//2, 2)+0.5
        break
    elif choice==2:
        data[:n//2, :2] = 0.5 * np.random.rand(n//2, 2)
        data[n//2:n, :2] = 0.6 * np.random.rand(n//2, 2)+0.4
        break
    elif choice==3:
        data[:n//2, :2] = 0.4 * np.random.rand(n//2, 2)
        data[n//2:3*n//4, 0:1] = 0.9 * np.random.rand(n//4, 1)
        data[n//2:3*n//4, 1:2] = 0.4 * np.random.rand(n//4, 1)+ 0.5
        data[3*n//4:n, 0:1] = 0.4 * np.random.rand(n//4, 1)+0.5
        data[3*n//4:n, 1:2] = 0.4 * np.random.rand(n//4, 1)
        break
    elif choice==4:
        data[:n//4, :2] = 0.4 * np.random.rand(n//4, 2)
        data[n//4:n//2, :2] = 0.4 * np.random.rand(n//4, 2) + 0.5
        data[n//2:3*n//4, 0:1] = 0.4 * np.random.rand(n//4, 1) + 0.5
        data[n//2:3*n//4, 1:2] = 0.4 * np.random.rand(n//4, 1) 
        data[3*n//4:n, 0:1] = 0.4 * np.random.rand(n//4, 1) 
        data[3*n//4:n, 1:2] = 0.4 * np.random.rand(n//4, 1) + 0.5
        break
    elif choice==5:
        r = 0.08*np.random.randn(n//2)
        theta = 2*np.pi*np.random.rand(n//2)
        data[:n//2, 0] = 0.5 + r*np.cos(theta)
        data[:n//2, 1] = 0.5 + r*np.sin(theta)
        
        r = 0.6+0.1*np.random.randn(n//2)
        theta = 2*np.pi*np.random.rand(n//2)
        data[n//2:, 0] = 0.5 + r*np.cos(theta)
        data[n//2:, 1] = 0.5 + r*np.sin(theta)
        break
    elif choice==0:
        break

y=np.zeros(n)

plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
plt.plot(w[:neyrones, 0], w[:neyrones, 1], 'ko', markersize=12, markerfacecolor='k')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Axis x')
plt.ylabel('Axis y')
plt.title(f'Protypa kai Epoch {epoch} kai Synapsewn {syek}')
epoch=epoch+1

ani = FuncAnimation(fig, animate, frames=120, interval=100,repeat=False)