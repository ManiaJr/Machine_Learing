# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

epoch=0
flag=True
times=1

def successful():
    recall = np.zeros((3, 2))
    recall[:, 0] = 0.4 * np.random.rand(3) + 0
    recall[:, 1] = 0.4 * np.random.rand(3) + 0.5

    if choice == 4:
        recall[1, 1] = 0.4 * np.random.rand() + 0

    recall[2:3, 0:2] = -1

    e = np.zeros((2,1))
    y1 = np.zeros((2,1))

    for i in range(2):
        e[i, 0] = np.dot(w.T, recall[:,i])
        if (e[i, 0] > 0):
            y1[i, 0] = 1
        else:
            y1[i, 0] = 0
            
        if (y1[i, 0] == 0):
            print(f'Elgxos me #{i+1} = {recall[:,i]} sthn klash 0(roz)')
        else:
            print(f'Elegxos me #{i+1} = {recall[:,i]} sthn klash 1(mayra)')

def animate(i):
    global epoch,flag,times
    while flag and epoch<=max_num_of_epochs:
        flag=False
        for i in range(n):
            u[i] = np.dot(data[i,:],w)
            if u[i]>0:
                v[i]=1
            else:
                v[i]=0
            if v[i]!=d[i]:
                for j in range(2):
                    w[j]=w[j]+ bhma * (d[i]-v[i])* data[i,j]
                flag=True
        for i in range(n):
            u[i] = np.dot(data[i,:],w)
            if u[i]>0:
                v[i]=1
            else:
                v[i]=0
        classA=np.where(v==0)
        classB=np.where(v==1)
        plt.figure(1)
        epoch+=1
        
        plt.subplot(2,2,2)
        plt.cla()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Train")
        plt.plot(data[classA,0],data[classA,1],"mo",data[classB,0],data[classB,1],"kd")
        
        
        x=data[:,0]
        y=-(w[0]*x -w[2])/w[1]
        plt.plot(x,y)
        plt.pause(0.05)
        
        
        plt.subplot(2,2,(3,4))
        plt.cla()
        plt.plot(np.arange(1,n//2+1),v[0:n//2],"mo",np.arange(n//2+1,n+1),v[n//2:n],"kd")
        plt.xlabel("Protypo")
        plt.ylabel("Exodos Y")
        plt.title("Protypa kai Exodoi")
        plt.pause(0.05)
        
        print(epoch)
    if flag!=True and times==1:
        successful()
        times+=1
#arxhh
fig=plt.figure()
plt.subplots_adjust(wspace=0.5,hspace=0.5)
n=int(input('\n Plhthos apo dedomena(pollaplasio toy 4 ): '))

data=np.zeros((n,2))
choice=1
while choice!=0:
    print("Epeleje ena apo ta parakatw: ")
    print("\n1.Grammika Diaxwrisima Protupa\n2.Mh Grammika Diaxwrisima Protupa (Klash 0 sth Gwnia)\n3.Mh Grammika Diaxwrisima Protupa (Klash 0 sto Kentro)\n4.Mh Grammika Diaxwrisima Protupa (XOR)\n5.Mh Grammika Diaxwrisima Protupa (Klash 0 mesa sthn Klash 1)\n0.Exit\n")
    choice=int(input("Epilogh: "))
    while choice<0 or choice>5:
        choise=input("Epilogh")
    if choice==1:
        data[:n//2, :2] = 0.4 * np.random.rand(n//2, 2)
        data[n//2:n, :2] = 0.4 * np.random.rand(n//2, 2)+0.5
        break
    elif choice==2:
        data[:n//2, :2] = 0.5 * np.random.rand(n//2, 2)
        data[n//2:n, :2] = 0.5 * np.random.rand(n//2, 2)+0.3
        break
    elif choice==3:
        data[:n//2, :2] = 0.4 * np.random.rand(n//2, 2)
        data[n//2:3*n//4, 0:1] = 0.9 * np.random.rand(n//4, 1)
        data[n//2:3*n//4, 1:2] = 0.4 * np.random.rand(n//4, 1)+ 0.5
        data[3*n//4:n, 0:1] = 0.5 * np.random.rand(n//4, 1)+0.5
        data[3*n//4:n, 1:2] = 0.5 * np.random.rand(n//4, 1)
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
        
        r = 0.4+0.1*np.random.randn(n//2)
        theta = 2*np.pi*np.random.rand(n//2)
        data[n//2:, 0] = 0.5 + r*np.cos(theta)
        data[n//2:, 1] = 0.5 + r*np.sin(theta)
        break
    elif choice==0:
        break
    
d=np.zeros(n)
d[n//2:n]=1
w = np.random.rand(3,1)
u = np.zeros(n)
v = np.zeros(n)
col=[[-1]*1]*n

data=np.append(data,col,axis=1)

bhma=float(input('\n bhma(0.01): '))
max_num_of_epochs=int(input('\n max apo epoxes: '))

plt.subplot(2,2,1)
plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'kd')
plt.xlabel('Axis x')
plt.ylabel('Axis y')
plt.title('Protypa')

ani = FuncAnimation(fig, animate, frames=120, interval=100,repeat=False)