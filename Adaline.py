# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

epoch=0
flag=True
times=1

def successful():
    recall = np.zeros((3,2))
    recall[:,0] = 0.4 * np.random.rand(3) + 0
    recall[:,1] = 0.4 * np.random.rand(3) + 0.5
    if choice == 4:
        recall[1,1] = 0.4 * np.random.rand(1) + 0
    recall[2:,0:2] = 1
    
    e = np.zeros((2,1))
    f = np.zeros((2,1))
    for i in range(2):
        e[i, 0] = np.dot(w.T, recall[:,i])
        f[i, 0] = e[i, 0]
        if f[i, 0] <= 0:
            print(f'Pattern #{i+1} = {recall[:,i]} in class -1')
        else:
            print(f'Pattern #{i+1} = {recall[:,i]} in class 1')

def animate(i):
    global epoch,flag,mse,times
    while flag and epoch<=max_num_of_epochs:
        for i in range(n):
            u = np.dot(data[i,:],w)
            v[i] = u
            delta[i]=d[i]-v[i]
            for j in range(3):
                w[j]=w[j]+bhma*delta[i]*data[i,j]
        sfalma=0
        for i in range(n):
            u = np.dot(data[i,:],w)
            v[i] = u
            delta[i]=d[i]-v[i]
            sfalma=sfalma+delta[i]**2
        if sfalma/n<=min_sfalma:
            flag=False
        epoch=epoch+1
        mse.insert(epoch, sfalma/n)
        plt.figure(1)
        classA=np.where(v<=0)
        classB=np.where(v>0)
        
        plt.subplot(2,2,1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'kd')
        plt.xlabel('Axis x')
        plt.ylabel('Axis y')
        plt.title('Protypa')
        plt.pause(0.05)
        
        plt.subplot(2,2,2)
        plt.cla()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot(data[classA,0],data[classA,1],"mo",data[classB,0],data[classB,1],"kd")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Train")
        
        x=data[:,0]
        y=-(w[0]*x -w[2])/w[1]
        plt.plot(x,y,'c')
        plt.pause(0.05)
        
        plt.subplot(2,2,3)
        plt.cla()
        v2=[1 if num>0 else -1 for num in v]
        plt.plot(np.arange(1,n//2+1),v2[0:n//2],"mo",np.arange(n//2+1,n+1),v2[n//2:n],"kd")
        plt.xlabel("Protypo")
        plt.ylabel("Exodos Y")
        plt.title("Protypa kai Exodoi")
        plt.pause(0.05)
        
        plt.subplot(2,2,4)
        plt.plot(mse,'c')
        plt.xlabel("Epoch")
        plt.ylabel("mse")
        plt.title("mse")
        plt.pause(0.05)
        
        print(epoch)
    if flag!=True and times==1:
        successful()
        times+=1
    
#arxhh
fig=plt.figure()
plt.subplots_adjust(wspace=0.5,hspace=0.5)
n=int(input('\n Plhthos apo dedomena(pollaplasio toy 8 ): '))

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
col=[[-1]*1]*n
data=np.append(data,col,axis=1)

d = np.ones((n, 1))
d[:n//2] = -1

ptp=np.dot(data.T,data)
pd=np.dot(data.T,d)

w = np.random.rand(3,1)
v = np.zeros(n)
mse=[]
delta = np.zeros(n)

bhma=float(input('\n bhma(0.01): '))
max_num_of_epochs=int(input('\n max apo epoxes: '))
#mporei na mhn einai float tha to doyme meta ayto 
min_sfalma=float(input("\n minimum sfalma:(0.1 peripou) "))

ani = FuncAnimation(fig, animate, frames=120, interval=100,repeat=False)